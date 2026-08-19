"""
Episodic K-sweep evaluation for the retrieval-based few-shot malware classifier.

Faithful to the protocol in QWEN_LM.py / README.md of this repo:
  - N-way K-shot episodes: sample N families, then K+5 disjoint vectors per
    family (K support, 5 query).
  - Fresh faiss.IndexFlatIP over the episode's support set (embeddings are
    L2-normalized, so inner product == cosine similarity).
  - Search with a fixed search-k of 10; empty (-1) slots are skipped, so the
    effective retrieval depth self-clamps to the support-pool size
    (e.g. 5 neighbors at N=5, K=1). This mirrors the original code exactly.
  - Prediction: similarity-weighted vote (sum of similarities per family,
    argmax; ties broken by retrieval-rank insertion order, as in the original).

What this script adds on top of QWEN_LM.py (without modifying it):
  - Seeded, reproducible episodes (default seeds 0-4, 500 episodes each).
  - N = 5 and N = 10 (the 5-vs-10 ambiguity is itself a finding).
  - 95% confidence intervals over episodes, plus per-seed means.
  - Stratification into three episode samplers:
      natural : families sampled uniformly (original protocol)
      hard    : episode forced to contain at least one known-confusable pair
      easy    : episode forced to contain NO complete confusable pair
  - Per-family query accuracy and an aggregate confusion matrix (from the
    natural episodes), so the "known-confusable" list can be checked against
    what the data actually confuses.
  - A synthetic smoke test (--smoke) so the pipeline can be validated without
    the real embeddings.

Inputs (either):
  --pt FILE          torch dict {family: [n, 2048] tensor} (repo format)
  --embeddings X.npy --labels Y.npy|Y.csv   flat (N_samples, D) + row labels

Outputs (per run, into --outdir):
  results_summary.csv, per_family_accuracy.csv, confusion_matrix.csv,
  confusion_top_pairs.csv, and raw per-episode accuracies (episodes.csv).

Usage:
  python episode_eval.py --pt qwen_pooled_malimg_emb.pt --outdir results_pooled
  python episode_eval.py --smoke --outdir results_smoke
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict

import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit("faiss-cpu is required: pip install faiss-cpu") from e

# ---------------------------------------------------------------------------
# Protocol constants (kept identical to QWEN_LM.py unless flagged otherwise)
# ---------------------------------------------------------------------------
QUERIES_PER_CLASS = 5          # as in QWEN_LM.py
SEARCH_K = 10                  # fixed search-k; -1 slots skipped (self-clamps)
DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_N_VALUES = (5, 10)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EPISODES = 500         # per (N, K, seed, stratum)

# Confusable pairs used for hard/easy stratification.
#
# These are DATA-DERIVED: top unordered pairs by symmetric confusion rate
# (row-normalized rate in both directions summed) from this replication's own
# aggregate confusion matrix over natural episodes (qwen_pooled embeddings).
# Notably they differ from the textbook MalImg list: Allaple.A's dominant
# confusion partner in this embedding space is Malex.gen!J (not Allaple.L),
# and Autorun.K / Yuner.A are NOT mutually confused (both ~1.0 accurate;
# Yuner.A instead absorbs Swizzor/C2LOP queries one-way).
# Override with --pairs "A:B,C:D" to stratify on a different list.
CONFUSABLE_PAIRS = [
    ("Allaple.A", "Malex.gen!J"),        # 0.187 symmetric rate
    ("Swizzor.gen!E", "Swizzor.gen!I"),  # 0.179
    ("C2LOP.gen!g", "Swizzor.gen!E"),    # 0.159
    ("Swizzor.gen!I", "Yuner.A"),        # 0.156 (one-way absorption)
    ("C2LOP.P", "C2LOP.gen!g"),          # 0.148
    ("Swizzor.gen!E", "Yuner.A"),        # 0.142 (one-way absorption)
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pt(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    fams = {k: v.to(torch.float32).numpy() for k, v in d.items()}
    return fams


def load_npy(emb_path, labels_path):
    X = np.load(emb_path).astype(np.float32)
    if labels_path.endswith(".npy"):
        labels = np.load(labels_path, allow_pickle=True).astype(str)
    else:  # csv: one label per row (optionally a 'family' column)
        with open(labels_path) as f:
            rows = list(csv.reader(f))
        if rows and rows[0] and any(c.lower() in ("family", "label") for c in rows[0]):
            col = [c.lower() for c in rows[0]].index(
                "family" if "family" in [c.lower() for c in rows[0]] else "label")
            labels = np.array([r[col] for r in rows[1:]])
        else:
            labels = np.array([r[0] for r in rows])
    assert len(labels) == len(X), "labels/embeddings row mismatch"
    fams = {}
    for fam in np.unique(labels):
        fams[str(fam)] = X[labels == fam]
    return fams


def make_smoke_data(n_families=25, per_family=120, dim=64, seed=123):
    """Gaussian clusters with two deliberately-confusable pairs."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_families, dim))
    # Make families 0/1 and 2/3 near-duplicates (confusable)
    centers[1] = centers[0] + rng.normal(scale=0.05, size=dim)
    centers[3] = centers[2] + rng.normal(scale=0.05, size=dim)
    fams = {}
    for i in range(n_families):
        pts = centers[i] + rng.normal(scale=0.35, size=(per_family, dim))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        fams[f"Fam{i:02d}"] = pts.astype(np.float32)
    return fams


def l2_normalize(fams):
    out = {}
    for k, v in fams.items():
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        out[k] = (v / n).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------
def contains_pair(families, pairs):
    fs = set(families)
    return any(a in fs and b in fs for a, b in pairs)


def sample_families(rng, all_families, n, stratum, pairs):
    """natural: uniform. hard: force one confusable pair in. easy: forbid all."""
    if stratum == "natural":
        return rng.sample(all_families, n)
    if stratum == "hard":
        usable = [p for p in pairs if p[0] in all_families and p[1] in all_families]
        a, b = usable[rng.randrange(len(usable))]
        rest = [f for f in all_families if f not in (a, b)]
        fams = [a, b] + rng.sample(rest, n - 2)
        rng.shuffle(fams)
        return fams
    if stratum == "easy":
        for _ in range(10000):
            fams = rng.sample(all_families, n)
            if not contains_pair(fams, pairs):
                return fams
        raise RuntimeError("could not sample an easy episode")
    raise ValueError(stratum)


def sample_episode(rng, fams_data, families, k):
    sup_vecs, sup_labels, qry_vecs, qry_labels = [], [], [], []
    for fam in families:
        mat = fams_data[fam]
        idx = rng.sample(range(len(mat)), k + QUERIES_PER_CLASS)
        sup_vecs.append(mat[idx[:k]])
        qry_vecs.append(mat[idx[k:]])
        sup_labels.extend([fam] * k)
        qry_labels.extend([fam] * QUERIES_PER_CLASS)
    return (np.concatenate(sup_vecs), sup_labels,
            np.concatenate(qry_vecs), qry_labels)


def run_episode(support, sup_labels, query, qry_labels, confusion=None,
                search_k=SEARCH_K):
    index = faiss.IndexFlatIP(support.shape[1])
    index.add(support)
    D, I = index.search(query, k=search_k)

    correct = 0
    for row in range(len(I)):
        scores = {}
        for j in range(len(I[row])):
            pos = I[row][j]
            if pos == -1:
                continue
            fam = sup_labels[pos]
            scores[fam] = scores.get(fam, 0.0) + float(D[row][j])
        pred = max(scores, key=scores.get)
        true = qry_labels[row]
        if confusion is not None:
            confusion[true][pred] += 1
        if pred == true:
            correct += 1
    return correct / len(qry_labels)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mean_ci95(vals):
    vals = np.asarray(vals, dtype=np.float64)
    m = vals.mean()
    if len(vals) < 2:
        return m, m, m
    sem = vals.std(ddof=1) / math.sqrt(len(vals))
    return m, m - 1.96 * sem, m + 1.96 * sem


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(fams_data, n_values, k_values, seeds, episodes, outdir, pairs,
              strata=("natural", "hard", "easy"), search_k_mode="fixed"):
    os.makedirs(outdir, exist_ok=True)
    all_families = sorted(fams_data.keys())
    max_k = max(k_values)
    for f in all_families:
        assert len(fams_data[f]) >= max_k + QUERIES_PER_CLASS, \
            f"{f} has too few samples for K={max_k}"

    episode_rows = []           # per-episode records
    summary_rows = []
    fam_correct = defaultdict(int)
    fam_total = defaultdict(int)
    confusion = {t: defaultdict(int) for t in all_families}

    for n in n_values:
        for k in k_values:
            # retrieval depth: 'fixed' = original protocol (always 10);
            # 'per-class' = k neighbors; 'full' = whole support pool (N*K)
            if search_k_mode == "fixed":
                sk = SEARCH_K
            elif search_k_mode == "per-class":
                sk = k
            elif search_k_mode == "full":
                sk = n * k
            else:
                raise ValueError(search_k_mode)
            for stratum in strata:
                per_seed_means = []
                accs_all = []
                for seed in seeds:
                    # str seeding is deterministic across processes (unlike
                    # hash() of a tuple containing str, which is salted)
                    rng = random.Random(f"N{n}-K{k}-{stratum}-seed{seed}")
                    accs = []
                    for ep in range(episodes):
                        families = sample_families(rng, all_families, n, stratum, pairs)
                        sup, sl, qry, ql = sample_episode(rng, fams_data, families, k)
                        conf = confusion if stratum == "natural" else None
                        acc = run_episode(sup, sl, qry, ql, confusion=conf,
                                          search_k=sk)
                        # per-family bookkeeping from natural episodes only
                        if stratum == "natural":
                            pass  # confusion already captures per-family stats
                        accs.append(acc)
                        episode_rows.append((n, k, stratum, seed, ep, acc,
                                             int(contains_pair(families, pairs))))
                    per_seed_means.append(float(np.mean(accs)))
                    accs_all.extend(accs)
                m, lo, hi = mean_ci95(accs_all)
                summary_rows.append({
                    "n_way": n, "k_shot": k, "stratum": stratum,
                    "mean_acc": round(m, 4), "ci95_lo": round(lo, 4),
                    "ci95_hi": round(hi, 4), "n_episodes": len(accs_all),
                    "seed_means": ";".join(f"{s:.4f}" for s in per_seed_means),
                    "seed_std": round(float(np.std(per_seed_means, ddof=1)), 4),
                })
                print(f"N={n} K={k:>2} {stratum:>7}: "
                      f"{m:.4f} [{lo:.4f}, {hi:.4f}]  seeds={per_seed_means}")

    # ---- write outputs -----------------------------------------------------
    with open(os.path.join(outdir, "results_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with open(os.path.join(outdir, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_way", "k_shot", "stratum", "seed", "episode",
                    "accuracy", "contains_confusable_pair"])
        w.writerows(episode_rows)

    # per-family accuracy from the natural-episode confusion matrix
    with open(os.path.join(outdir, "per_family_accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_queries", "accuracy"])
        for fam in all_families:
            total = sum(confusion[fam].values())
            corr = confusion[fam][fam]
            w.writerow([fam, total, round(corr / total, 4) if total else ""])

    with open(os.path.join(outdir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + all_families)
        for t in all_families:
            w.writerow([t] + [confusion[t][p] for p in all_families])

    # top confused (true, pred) off-diagonal pairs, as row-normalized rates
    top = []
    for t in all_families:
        total = sum(confusion[t].values())
        if not total:
            continue
        for p, c in confusion[t].items():
            if p != t and c > 0:
                top.append((c / total, c, t, p))
    top.sort(reverse=True)
    with open(os.path.join(outdir, "confusion_top_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "count", "rate_of_true_queries"])
        for rate, c, t, p in top[:30]:
            w.writerow([t, p, c, round(rate, 4)])

    print(f"\nWrote results to {outdir}/")
    return summary_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pt", help="torch dict {family: [n,D] tensor} (repo format)")
    ap.add_argument("--embeddings", help="embeddings.npy, shape (N, D)")
    ap.add_argument("--labels", help="labels.npy or .csv, one family per row")
    ap.add_argument("--smoke", action="store_true",
                    help="run on synthetic Gaussian clusters (no files needed)")
    ap.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                    help="episodes per (N, K, seed, stratum)")
    ap.add_argument("--outdir", default="results_episode_eval")
    ap.add_argument("--strata", nargs="+",
                    default=["natural", "hard", "easy"],
                    choices=["natural", "hard", "easy"])
    ap.add_argument("--search-k-mode", default="fixed",
                    choices=["fixed", "per-class", "full"],
                    help="fixed=search-k 10 as in QWEN_LM.py; per-class=K; "
                         "full=N*K (whole support pool)")
    ap.add_argument("--pairs", default=None,
                    help='override confusable pairs, e.g. "A:B,C:D"')
    args = ap.parse_args()

    if args.smoke:
        fams = make_smoke_data()
        pairs = [("Fam00", "Fam01"), ("Fam02", "Fam03")]
    elif args.pt:
        fams = load_pt(args.pt)
        pairs = CONFUSABLE_PAIRS
    elif args.embeddings and args.labels:
        fams = load_npy(args.embeddings, args.labels)
        pairs = CONFUSABLE_PAIRS
    else:
        ap.error("provide --pt, or --embeddings + --labels, or --smoke")

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]

    # NOT merely cosmetic: the .pt embeddings are stored in bfloat16, which
    # leaves vector norms jittering in [0.9964, 1.0036]. Raw inner-product
    # retrieval (as in QWEN_LM.py) multiplies that jitter into the scores, and
    # because neighbor similarities here are packed within ~1e-3 of each other,
    # it reorders neighbors: measured effect is ~ -10 pts at 1-shot on the
    # pooled embeddings and ~ -32 pts on the lastpooled ones. Re-normalizing
    # in float32 restores true cosine ranking.
    fams = l2_normalize(fams)
    run_sweep(fams, args.n_values, args.k_values, args.seeds,
              args.episodes, args.outdir, pairs,
              strata=args.strata, search_k_mode=args.search_k_mode)


if __name__ == "__main__":
    main()
