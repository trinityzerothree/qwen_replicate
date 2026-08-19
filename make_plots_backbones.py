"""Four backbone-comparison plots (reads results_*/results_summary.csv)."""
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"
ORANGE_LT = "#f5a37c"   # lighter step of the orange for the within-family pair
AQUA_LT = "#7dd0ae"     # lighter step of the aqua

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True,
})

K = [1, 3, 5, 10]


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["stratum"] == "natural" and int(r["n_way"]) == 5:
            out[int(r["k_shot"])] = (float(r["mean_acc"]),
                                     float(r["ci95_lo"]), float(r["ci95_hi"]))
    return out


MODELS = [
    # key, dir, raw dir, color, marker, label
    ("clip",      "results_clip",        "results_clip_raw",            BLUE,      "o", "CLIP ViT-B/32"),
    ("dino_cls",  "results_dino_cls",    "results_dino_cls_raw",        ORANGE,    "o", "DINOv2-base (CLS)"),
    ("dino_mean", "results_dino_noncls", "results_dino_noncls_raw",     ORANGE_LT, "s", "DINOv2-base (mean)"),
    ("qwen_pool", "results_pooled",      "results_qwen_pooled_raw",     AQUA,      "o", "Qwen2.5-VL 3B (mean pool)"),
    ("qwen_last", "results_lastpooled",  "results_qwen_lastpooled_raw", AQUA_LT,   "s", "Qwen2.5-VL 3B (last token)"),
]
res = {m[0]: load(f"{m[1]}/results_summary.csv") for m in MODELS}
raw = {m[0]: load(f"{m[2]}/results_summary.csv") for m in MODELS}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(1.0)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.set_xticks(K); ax.tick_params(length=0)


def draw_line(ax, key, color, marker, label):
    m = np.array([res[key][k][0] for k in K])
    lo = np.array([res[key][k][1] for k in K])
    hi = np.array([res[key][k][2] for k in K])
    ax.fill_between(K, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(K, m, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", label=label, zorder=3)
    ax.plot(K, m, marker, markersize=8, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    return m

# ------------------------------------------------------------ 1. K-sweep
fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
fig.patch.set_facecolor(SURFACE)
style_ax(ax)
ends = {}
for key, _, _, color, marker, label in MODELS:
    ends[key] = draw_line(ax, key, color, marker, label)

# direct labels at the right edge, cluster-style
lab = [
    ("dino_mean", "DINOv2 (mean)", 10),
    ("dino_cls",  "DINOv2 (CLS)", -6),
    ("clip",      "CLIP", -20),
    ("qwen_pool", "Qwen (mean pool)", 6),
    ("qwen_last", "Qwen (last token)", -8),
]
for key, txt, dy in lab:
    ax.annotate(txt, (10, ends[key][-1]), xytext=(12, dy),
                textcoords="offset points", color=INK2, fontsize=10, va="center")

ax.set_xlim(0.4, 12.6)
ax.set_ylim(0.75, 1.0)
ax.set_yticks(np.arange(0.75, 1.001, 0.05))
ax.set_xlabel("K (shots per class)", fontsize=11)
ax.set_ylabel("Episode accuracy", fontsize=11)
ax.set_title("Small self-supervised backbones beat the 3B VLM at every K",
             fontsize=15, color=INK, loc="left", pad=18, fontweight="semibold")
ax.text(0, 1.015, "5-way natural episodes · 500 episodes × 5 seeds · 95% CI bands · pooling choice barely moves either family; the K=5 dip appears in every line",
        transform=ax.transAxes, color=INK2, fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plot1_ksweep_5models.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 2. gap vs K
def sem(key, k, table):
    m, lo, hi = table[key][k]
    return (hi - m) / 1.96

best_qwen = {k: max(res["qwen_pool"][k][0], res["qwen_last"][k][0]) for k in K}
best_key = {k: ("qwen_pool" if res["qwen_pool"][k][0] >= res["qwen_last"][k][0]
                else "qwen_last") for k in K}

fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
style_ax(ax)
for key, color, marker, label in [
        ("clip", BLUE, "o", "CLIP ViT-B/32"),
        ("dino_cls", ORANGE, "o", "DINOv2-base (CLS)"),
        ("dino_mean", ORANGE_LT, "s", "DINOv2-base (mean)")]:
    gap = np.array([(res[key][k][0] - best_qwen[k]) * 100 for k in K])
    ci = np.array([1.96 * math.hypot(sem(key, k, res),
                                     sem(best_key[k], k, res)) * 100 for k in K])
    ax.fill_between(K, gap - ci, gap + ci, color=color, alpha=0.12, linewidth=0)
    ax.plot(K, gap, color=color, linewidth=2, solid_capstyle="round", label=label)
    ax.plot(K, gap, marker, markersize=8, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    for k, g in zip(K, gap):
        if k in (1, 10) and key == "dino_cls":
            dx, dy = (0, 14) if k == 1 else (-22, 6)
            ax.annotate(f"+{g:.1f}", (k, g), xytext=(dx, dy),
                        textcoords="offset points", ha="center",
                        color=INK2, fontsize=10.5)

ax.axhline(0, color=BASELINE, linewidth=1.0)
ax.text(10.55, 0.3, "parity with Qwen", color=MUTED, fontsize=9.5, va="bottom")
ax.set_xlim(0.4, 11.4)
ax.set_ylim(-1, 14)
ax.set_xlabel("K (shots per class)", fontsize=11)
ax.set_ylabel("Accuracy gap vs best Qwen variant (points)", fontsize=11)
ax.set_title("The encoder deficit is largest exactly in the low-K regime the paper targets",
             fontsize=15, color=INK, loc="left", pad=18, fontweight="semibold")
ax.text(0, 1.015, "Backbone accuracy minus best Qwen2.5-VL variant at each K · 5-way natural episodes · CI: propagated 95%",
        transform=ax.transAxes, color=INK2, fontsize=10)
ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plot2_gap_vs_k.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------- 3. before/after normalization
fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ax.set_axisbelow(True); ax.tick_params(length=0)

names = ["CLIP\nViT-B/32", "DINOv2\n(CLS)", "DINOv2\n(mean)",
         "Qwen 3B\n(mean pool)", "Qwen 3B\n(last token)"]
keys = ["clip", "dino_cls", "dino_mean", "qwen_pool", "qwen_last"]
before = [raw[k][1][0] for k in keys]
after = [res[k][1][0] for k in keys]
x = np.arange(5)
w = 0.22
b1 = ax.bar(x - w / 2 - 0.015, before, w, color=MUTED,
            label="raw inner product (as cached)", zorder=3)
b2 = ax.bar(x + w / 2 + 0.015, after, w, color=BLUE,
            label="float32 re-normalized", zorder=3)
for bars in (b1, b2):
    for r in bars:
        r.set_linewidth(0)
for i, (b, a) in enumerate(zip(before, after)):
    d = (a - b) * 100
    txt = f"+{d:.1f}" if d > 0.05 else "±0.0"
    ax.annotate(txt, (x[i], max(a, b)), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK2,
                fontsize=11, fontweight="bold" if d > 0.05 else "normal")

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10, color=INK2)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("1-shot episode accuracy (5-way)", fontsize=11)
ax.set_title("The normalization bug was Qwen-cache-specific, not a universal bf16 problem",
             fontsize=15, color=INK, loc="left", pad=18, fontweight="semibold")
ax.text(0, 1.02, "CLIP/DINO caches are float32 with exact unit norms — identical episodes, delta exactly 0 · Qwen caches are bfloat16 (norms 0.996–1.004)",
        transform=ax.transAxes, color=INK2, fontsize=10)
ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK2)
fig.tight_layout()
fig.savefig("plot3_normalization_control.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------ 4. accuracy vs parameter count
pts = [
    ("CLIP ViT-B/32", 87.5e6, res["clip"][1][0], BLUE),
    ("DINOv2-base", 86.6e6, max(res["dino_cls"][1][0], res["dino_mean"][1][0]), ORANGE),
    ("Qwen2.5-VL 3B", 3.75e9, max(res["qwen_pool"][1][0], res["qwen_last"][1][0]), AQUA),
]
fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_xscale("log")

for name, p, a, c in pts:
    ax.plot(p, a, "o", markersize=13, markerfacecolor=c,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
offsets = {"CLIP ViT-B/32": (-72, -14), "DINOv2-base": (10, 16),
           "Qwen2.5-VL 3B": (0, 16)}
for name, p, a, c in pts:
    dx, dy = offsets[name]
    ax.annotate(f"{name}\n{a:.3f}", (p, a), xytext=(dx, dy),
                textcoords="offset points", ha="center", color=INK2, fontsize=10.5)

ax.annotate("43× the parameters,\n11–12 points worse",
            xy=(3.75e9, pts[2][2]), xytext=(-130, -40), textcoords="offset points",
            color=INK2, fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=1))

ax.set_xlim(4e7, 1.2e10)
ax.set_ylim(0.75, 0.96)
ax.set_xticks([1e8, 1e9, 1e10])
ax.set_xticklabels(["100M", "1B", "10B"])
ax.set_xlabel("Backbone parameters (log scale)", fontsize=11)
ax.set_ylabel("1-shot episode accuracy (5-way)", fontsize=11)
ax.set_title("More parameters, worse malware retrieval",
             fontsize=15, color=INK, loc="left", pad=18, fontweight="semibold")
ax.text(0, 1.015, "Best pooling variant per model · 5-way 1-shot, natural episodes, 2,500 episodes per point",
        transform=ax.transAxes, color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig("plot4_params_vs_accuracy.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("wrote plot1..plot4")
