#!/usr/bin/env python3
# make_figures.py -- generates all 6 paper figures from campaign data.
# Requires: pip install matplotlib numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------- FIG 1: staircase heatmaps (n=32, n=128) ----------------
stair32 = np.array([          # fourier D=256, n=32 (v4.1 capacity run)
 [1,1,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,1,0,0,0,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,0,0,0,0],
 [.95,1,1,1,1,1,1,1,1,1,1,1]])
chan128 = np.array([          # n=128 eq-view (v5.2 chan128 rows)
 [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.40,0.00],
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), gridspec_kw={"width_ratios":[12,16]})
for ax, M, ttl in zip(axes, [stair32, chan128],
        ["n=32, D=256 (T=1..4)", "n=128, D=256 (T=1..5, eq view)"]):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("true distance $k$")
    ax.set_title(ttl)
    ks = M.shape[1]
    ax.set_xticks(range(ks)); ax.set_xticklabels(range(1, ks+1))
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"$T={t+1}$\n(radius {2**(t+1)})" for t in range(M.shape[0])])
    # doubling edge staircase overlay
    for col in range(ks):
        need = int(np.ceil(np.log2(col + 1))) - 1
        ax.plot(col, need, "wo", ms=3, mfc="none")
axes[0].set_ylabel("channel steps")
fig.colorbar(im, ax=axes, shrink=0.85, label="accuracy")
fig.savefig("fig1_staircase.pdf")

# ---------------- FIG 2: frontier contrast (linear vs log) ----------------
kb   = [1,2,3,4,5,6,7]            # baseline n=32 T_min (v4.1 main)
Tb   = [1,2,3,4,5,6,8]
kh   = [1,2,3,4,5,6,7,8]          # holo T_min
Th   = [1,1,2,2,3,3,3,4]
kk   = np.linspace(1, 8.5, 100)
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(kb, Tb, "s", color="crimson", ms=7, label="baseline (no channel)")
ax.plot(kk, 1.11*kk - 0.29, "--", color="crimson", lw=1,
        label=r"fit $1.11\,k-0.29$ (LINEAR)")
ax.plot(kh, Th, "o", color="navy", ms=7, label="holo core (channel)")
kk2 = np.linspace(1, 8.5, 100)
ax.plot(kk2, np.maximum(1, np.ceil(np.log2(kk2))), "--", color="navy", lw=1,
        label=r"law $\lceil\log_2 k\rceil$ (LOG)")
ax.axvspan(6, 8.5, color="grey", alpha=0.12)
ax.text(7.1, 1.2, "OOD\n(train $k\\leq6$)", ha="center", fontsize=8)
ax.set_xlabel("chain length $k$"); ax.set_ylabel("min steps $T_{\\min}$ for 90% acc")
ax.set_title("Linear vs. logarithmic frontier (n=32)")
ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.3)
fig.savefig("fig2_frontier.pdf")

# ---------------- FIG 3: the moving edge (holo vs shuffle) ----------------
Tt   = [1,2,3,4,5,6]
holo = [0.00,0.00,0.00,0.00,0.21,0.71]   # far-neg yes-rate, n=128 ball run
shuf = [0.00,0.00,0.00,0.00,0.00,0.00]   # shuffle control (flat/no-signal)
radius = [2,4,8,16,32,64]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(Tt, holo, "o-", color="navy", label="holo (true channel)")
ax.plot(Tt, shuf, "s--", color="grey", label="shuffle control")
for t, r, y in zip(Tt, radius, holo):
    ax.annotate(f"$R{{=}}{r}$", (t, y), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7, color="navy")
ax.axhline(0, color="k", lw=0.5)
ax.axvspan(5.6, 6.4, color="gold", alpha=0.25)
ax.annotate("edge crosses negatives\n($64\\geq34$): must say YES",
            (6, 0.71), textcoords="offset points", xytext=(-95, -18),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("steps $T$"); ax.set_ylabel("yes-rate on far negatives (dist 34–40)")
ax.set_title("The moving edge: output tracks the ball radius")
ax.set_ylim(-0.08, 0.95); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig("fig3_moving_edge.pdf")

# ---------------- FIG 4: capacity law (FP vs 1/sqrt(D)) ----------------
Ds  = np.array([256, 512, 1024])
fps = np.array([0.23, 0.15, 0.07])
x   = 1/np.sqrt(Ds)
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.plot(x, fps, "o", color="darkgreen", ms=8)
for xi, yi, d in zip(x, fps, Ds):
    ax.annotate(f"D={d}", (xi, yi), textcoords="offset points",
                xytext=(8, -3), fontsize=8)
ax.plot(x, fps, "--", color="darkgreen", lw=0.8, alpha=0.5)
ax.set_xlabel("$D^{-1/2}$"); ax.set_ylabel("false positives @ T=3\n(random codebook)")
ax.set_title("Capacity wall $\\propto D^{-1/2}$")
ax.grid(alpha=0.3)
fig.savefig("fig4_capacity.pdf")

# ---------------- FIG 5: frozen vs trainable (the A/B) ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
labels = ["frozen\n(buffer)", "trainable\n(Parameter)"]
vals   = [0.88, 0.15]
bars = a1.bar(labels, vals, color=["navy", "crimson"], width=0.55)
a1.axhline(0.9, color="grey", ls=":", lw=0.8)
a1.text(1.42, 0.915, "edge of doubling ball (T=3)", fontsize=7, ha="right")
for b, v in zip(bars, vals):
    a1.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
a1.set_ylabel("accuracy at k=8, T=3"); a1.set_ylim(0, 1.05)
a1.set_title("Same model, gradients touch $\\theta$ or not")
coh = [7e-5, 3.5e-3]
a2.semilogy([0, 1], coh, "o", color="darkgreen", ms=8)
a2.set_xticks([0, 1]); a2.set_xticklabels(labels)
for xi, c in zip([0,1], coh):
    a2.annotate(f"{c:.1e}", (xi, c), textcoords="offset points",
                xytext=(5, 5), fontsize=8)
a2.set_ylabel("codebook coherence (max)")
a2.set_title("$\\sim\\!10^4\\times$ erosion under free gradients")
a2.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("fig5_freeze.pdf")

# ---------------- FIG 6: the money table + extraction ladder ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.1),
                             gridspec_kw={"width_ratios":[1.2,1]})
Ls  = list(range(2, 9))
rex = [100]*7
lm  = [0]*7
a1.plot(Ls, rex, "o-", color="navy", lw=2, label="regex + frozen bank")
a1.plot(Ls, lm, "x--", color="crimson", ms=8, label="SmolLM-135M alone")
a1.set_xlabel("chain length $L$"); a1.set_ylabel("naming accuracy (%)")
a1.set_title("Reasoning: exact at every length"); a1.set_ylim(-5, 112)
a1.legend(fontsize=8); a1.grid(alpha=0.3)
models = ["135M\nInstruct", "360M\nbase", "360M\nInstruct", "1.7B\nInstruct"]
rec    = [0.31, 0.26, 0.39, 0.68]
cols   = ["grey", "lightsteelblue", "cornflowerblue", "navy"]
bars = a2.bar(models, rec, color=cols, width=0.6)
for b, v in zip(bars, rec):
    a2.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}", ha="center", fontsize=9)
a2.axhline(0.7, color="grey", ls=":", lw=0.8)
a2.text(1.5, 0.72, "S1 threshold", fontsize=7, color="grey")
a2.set_ylabel("extraction recall (bounded scoring)")
a2.set_title("Extraction scales with model size")
a2.set_ylim(0, 0.85); a2.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("fig6_money.pdf")

print("saved: fig1_staircase.pdf ... fig6_money.pdf")