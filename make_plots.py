"""Slide plots for the episodic K-sweep evaluation (reads results_*/ CSVs)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette / chrome (dataviz reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
})

K_VALUES = [1, 3, 5, 10]
PAPER_TABLE2 = {1: 0.878, 3: 0.875, 5: 0.864, 10: 0.886}  # paper, N=5


def load_summary(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        key = (int(r["n_way"]), int(r["k_shot"]), r["stratum"])
        out[key] = (float(r["mean_acc"]), float(r["ci95_lo"]), float(r["ci95_hi"]))
    return out


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(K_VALUES)
    ax.tick_params(length=0)


def series(ax, res, n, stratum, color, label, direct_label=None, dy=0):
    m = np.array([res[(n, k, stratum)][0] for k in K_VALUES])
    lo = np.array([res[(n, k, stratum)][1] for k in K_VALUES])
    hi = np.array([res[(n, k, stratum)][2] for k in K_VALUES])
    ax.fill_between(K_VALUES, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(K_VALUES, m, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", label=label, zorder=3)
    ax.plot(K_VALUES, m, "o", markersize=8, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    if direct_label:
        ax.annotate(direct_label, (K_VALUES[-1], m[-1]),
                    xytext=(10, dy), textcoords="offset points",
                    color=INK2, fontsize=10.5, va="center")
    return m


# ---------------------------------------------------------------- slide 1
res = load_summary("results_pooled/results_summary.csv")

fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
style_ax(ax)

m5 = series(ax, res, 5, "natural", BLUE, "5-way (this replication)",
            "5-way", dy=6)
m10 = series(ax, res, 10, "natural", ORANGE, "10-way (this replication)",
             "10-way", dy=0)

# paper's Table-2 numbers (5-way, ~5 episodes) as muted reference markers
pk = list(PAPER_TABLE2.keys())
pv = list(PAPER_TABLE2.values())
ax.plot(pk, pv, "s", markersize=7, markerfacecolor="none",
        markeredgecolor=MUTED, markeredgewidth=1.6,
        label="paper Table 2 (5-way, few episodes)", zorder=3)

# selective direct labels: the dip point and the 10-shot endpoint
ax.annotate(f"{m5[2]:.3f}", (5, m5[2]), xytext=(0, -18),
            textcoords="offset points", ha="center", color=INK2, fontsize=10.5)
ax.annotate(f"{m5[3]:.3f}", (10, m5[3]), xytext=(0, 12),
            textcoords="offset points", ha="center", color=INK2, fontsize=10.5)

ax.set_xlim(0.4, 11.4)
ax.set_ylim(0.60, 1.0)
ax.set_yticks(np.arange(0.60, 1.01, 0.05))
ax.set_xlabel("K (shots per class)", fontsize=11)
ax.set_ylabel("Episode accuracy", fontsize=11)
ax.set_title("The 5-shot dip survives 2,500 episodes — it is protocol, not noise",
             fontsize=15, color=INK, loc="left", pad=16, fontweight="semibold")
ax.text(0, 1.015, "Qwen2.5-VL retrieval classifier on MalImg · 500 episodes × 5 seeds per point · shaded bands: 95% CI",
        transform=ax.transAxes, color=INK2, fontsize=10.5)
ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig("slide1_ksweep.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- slide 2
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), dpi=160, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, n in zip(axes, (5, 10)):
    style_ax(ax)
    series(ax, res, n, "easy", BLUE, "easy episodes (no confusable pair)",
           "easy" if n == 10 else None, dy=4)
    series(ax, res, n, "natural", MUTED, "natural sampling",
           "natural" if n == 10 else None, dy=0)
    series(ax, res, n, "hard", ORANGE, "hard episodes (confusable pair forced in)",
           "hard" if n == 10 else None, dy=-4)
    ax.set_xlim(0.4, 11.4)
    ax.set_ylim(0.60, 1.0)
    ax.set_title(f"{n}-way", fontsize=12.5, color=INK2, loc="left")
    ax.set_xlabel("K (shots per class)", fontsize=11)
axes[0].set_ylabel("Episode accuracy", fontsize=11)
axes[0].legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)

fig.suptitle("A dozen points of headline accuracy is family composition, not the method",
             fontsize=15, color=INK, x=0.045, y=1.00, ha="left", fontweight="semibold")
fig.text(0.045, 0.945,
         "Episodes stratified on the replication's own top confusable pairs "
         "(Allaple.A↔Malex.gen!J, Swizzor E↔I, C2LOP cluster, Swizzor→Yuner.A) · 95% CI bands",
         color=INK2, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("slide2_stratification.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- bonus: ablation slide
res_pc = load_summary("results_ablation_perclass/results_summary.csv")
res_full = load_summary("results_ablation_full/results_summary.csv")

fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
style_ax(ax)

series(ax, res, 5, "natural", ORANGE, "fixed search-k = 10 (paper protocol)",
       "fixed k=10", dy=8)
series(ax, res_pc, 5, "natural", BLUE, "search-k = K (per-class clamp)",
       "k = K", dy=-2)
mfull = series(ax, res_full, 5, "natural", MUTED, "search-k = N·K (whole support pool)",
               "k = N·K", dy=-14)

ax.set_xlim(0.4, 11.6)
ax.set_ylim(0.75, 0.95)
ax.set_yticks(np.arange(0.75, 0.951, 0.05))
ax.set_xlabel("K (shots per class)", fontsize=11)
ax.set_ylabel("Episode accuracy", fontsize=11)
ax.set_title("The dip is the fixed retrieval depth: change search-k and it disappears",
              fontsize=15, color=INK, loc="left", pad=16, fontweight="semibold")
ax.text(0, 1.015, "5-way, natural episodes, same seeds/episodes as the main sweep · at K=5 a single wrong family can fill 5 of 10 vote slots",
        transform=ax.transAxes, color=INK2, fontsize=10.5)
ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig("slide3_searchk_ablation.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("wrote slide1_ksweep.png, slide2_stratification.png, slide3_searchk_ablation.png")
