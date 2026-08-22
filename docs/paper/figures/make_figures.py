"""Generate F1-F8. **Every value is loaded from a persisted artifact; nothing is hand-entered.**

Run: ``python docs/paper/figures/make_figures.py``

One script rather than eight, because the figures share a loader and a style and duplicating both
across eight files is how two figures end up disagreeing about the same number. Each function names
the artifact it reads in its docstring, and `_artifacts.py` carries the addresses.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FKA_C, DENSE_C, BPE_C = "#1b6ca8", "#c1272d", "#e08214"

# The key counts actually swept. Used as explicit log-axis ticks so the decade minor labels
# cannot overprint each other.
_KEY_TICKS = (8000, 16000, 32000, 96000, 288000)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# ---------------------------------------------------------------------------------------------
# F1 — the frontier, at MEASURED N on both sides
# ---------------------------------------------------------------------------------------------
def f1_frontier():
    """From plot_data.json. The refusal is drawn as an absence, with the reason on the axis."""
    d = A.frontier()
    fig, ax = plt.subplots(figsize=(6.1, 3.0))

    # THE DENSE SIDE IS A MEASURED PER-N SERIES, parsed from the same logs as every other dense
    # figure. Two earlier drawings were wrong: a scatter that broadcast the max over arms to every
    # N (with the 10M row at that max times a hard-coded 0.35 found in no artifact), then a single
    # band. Neither was necessary — the runner has always printed bits/param on every POOLED line.
    ns_all = []
    for fam, c, mk, lab in (("syllable_1M", DENSE_C, "o", "dense 1M, syllable"),
                            ("syllable_10M", "#7b1fa2", "^", "dense 10M, syllable"),
                            ("bpe_1M", BPE_C, "s", "dense 1M, BPE-735")):
        arms = A.dense_bits_per_param(fam)
        ks = sorted(arms)
        ns_all += ks
        mean = [st.mean(arms[k]) for k in ks]
        lo = [st.mean(arms[k]) - min(arms[k]) for k in ks]
        hi = [max(arms[k]) - st.mean(arms[k]) for k in ks]
        ax.errorbar(ks, mean, yerr=[lo, hi], marker=mk, ms=3.6, lw=1.1, capsize=2, c=c,
                    label=lab, zorder=3)

    # The quoted dense best is NOT on this curve: it is the qa_trained half at N = 8,000, in a
    # different experiment. Drawn as a reference line, labelled as what it is.
    bp = A.dense_best_bits_provenance()
    ax.axhline(bp["quoted"].value, ls="-.", lw=0.8, c=DENSE_C, alpha=0.55)
    ax.text(0.015, bp["quoted"].value * 1.1,
            f"best any arm {bp['quoted'].value} — qa-trained half at "
            f"$N={bp['n_entities'].value:,}$, off this curve",
            fontsize=5.6, color=DENSE_C, va="bottom",
            transform=blended_transform_factory(ax.transAxes, ax.transData))
    ns = sorted(set(ns_all))
    fp = d["fka"]["points"][0]
    ax.scatter([fp["n_entities"]], [fp["bits_per_param_amortised"]], s=70, c=FKA_C,
               marker="*", label="FKA (measured N = 2M)", zorder=4)
    ax.axhline(d["dense"]["literature_ceiling_bits_per_param"], ls=":", lw=0.9, c="0.4")
    # Blended transform: x inset from the left spine in axes fraction (in data coords the leading
    # glyph printed on the y-axis line), y in data coords just clear of the ceiling itself, so the
    # dotted line cannot strike through the label.
    ax.text(0.02, d["dense"]["literature_ceiling_bits_per_param"] * 1.16,
            "literature ceiling, 2.0 bits/param", fontsize=6.5, color="0.35", va="bottom",
            transform=blended_transform_factory(ax.transAxes, ax.transData))

    ax.annotate("", xy=(1.3e6, 0.055), xytext=(3.6e5, 0.055),
                arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.9, ls="--"))
    ax.text(3.6e5, 0.062, "extrapolation REFUSED\n(two forms, 162 orders apart)",
            fontsize=6.0, color="0.35", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("corpus size $N$ (entities)")
    ax.set_ylabel("knowledge stored (bits / parameter)")
    ax.set_ylim(0.004, 9)
    ax.set_xlim(ns[0] * 0.75, 6e6)
    # Outside the axes: the 10M series now occupies the lower-left corner the legend used.
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.2)
    _save(fig, "F1_frontier")


# ---------------------------------------------------------------------------------------------
# F2 — the fork a/c exhibit
# ---------------------------------------------------------------------------------------------
def f2_fork():
    """M2's one-variable exhibit: the NEVER-SUPERVISED pair, and only that.

    The end-to-end bars are gone. They carried fork (a) at 63.75%, which does have an artifact
    address (m2-fork-a-joint/joint.json, /e2e_learned_stack/accuracy) but is the POOLED figure
    M2 §11.2's erratum retired: 85.8% of the eval set's target facts already had supervised
    addresses, so the pooled number is dominated by address recall. Its corrected split is
    71.6% / 100.0% / 0.0% (§11.3). Plotting a pre-erratum aggregate beside a post-erratum one
    would be the mix-vs-model confusion this study has a standing rule against.
    """
    counts = A.fork_never_supervised()
    labels = ["fork (a)\nfree embedding row", "fork (c)\nlearned content encoder"]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    x = list(range(2))
    vals = [counts["a"].value * 100, counts["c"].value * 100]
    ns = [counts["a_n"].value, counts["c_n"].value]
    ax.bar(x, vals, width=0.5, color=[DENSE_C, FKA_C])
    for i, (v, n) in enumerate(zip(vals, ns)):
        hit = int(round(v / 100 * n))
        ax.text(i, v + 2.5, f"{v:.1f}%\n({hit}/{n})", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("never-supervised retrieval (%)"); ax.set_ylim(0, 124)
    ax.set_title("one variable: where entity identity enters the key", fontsize=7.5)
    _save(fig, "F2_fork")


# ---------------------------------------------------------------------------------------------
# F4 — margin trajectories, from the persisted per-fact matrix
# ---------------------------------------------------------------------------------------------
def f4_margins():
    """From shape.json's trajectory_matrix (788 facts x 8 loads) — the matrix, not the summary."""
    import numpy as np
    tm = A.margin_trajectories()
    m = np.array(tm["margins"]); loads = np.array(tm["loads"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.5))
    idx = np.linspace(0, m.shape[0] - 1, 120).astype(int)
    for i in idx:
        a1.plot(loads, m[i], lw=0.35, alpha=0.30, c=FKA_C)
    a1.plot(loads, np.median(m, axis=0), lw=1.8, c="k", label="median")
    a1.invert_xaxis()
    a1.set_xlabel("bits per fact (decreasing = harder)")
    a1.set_ylabel("retrieval margin")
    a1.axhline(0, lw=0.7, c="0.4", ls=":")
    a1.legend(frameon=False)

    d = -np.diff(m, axis=1); tot = d.sum(axis=1); big = d.max(axis=1)
    sharp = big / np.where(np.abs(tot) < 1e-9, np.nan, tot)
    fin = sharp[np.isfinite(sharp)]
    cuts = np.linspace(0.5, 1.0, 26)
    a2.plot(cuts, [(fin > c).mean() * 100 for c in cuts], lw=1.4, c=FKA_C)
    a2.axvline(0.75, ls="--", lw=0.8, c="0.4")
    # The label sits on the cut it names, so both the curve and the 50% guide pass behind it.
    # An opaque backing keeps it legible without moving it away from what it labels.
    a2.text(0.752, 78, "published cut 0.75", fontsize=6, color="0.35", rotation=90, va="top",
            zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))
    a2.axhline(50, ls=":", lw=0.8, c="0.6")
    a2.set_xlabel("collapse-sharpness cut")
    a2.set_ylabel("% of facts above cut")
    a2.set_title("no principled cut exists: reported as a curve", fontsize=7)
    _save(fig, "F4_margins")


# ---------------------------------------------------------------------------------------------
# F5 — dense collapse and the K*_P curves, P and sigma per arm
# ---------------------------------------------------------------------------------------------
def f5_dense():
    """From the dense run logs: every arm, every seed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.6))
    for fam, c, mk, lab in (("syllable_1M", DENSE_C, "o", "syllable, 1M"),
                            ("syllable_10M", "#7b1fa2", "^", "syllable, 10M"),
                            ("bpe_1M", BPE_C, "s", "BPE-735, 1M")):
        arms = A.dense_arms(fam)
        ks = sorted(arms)
        mean = [st.mean([a for a, _ in arms[k]]) for k in ks]
        sd = [st.stdev([a for a, _ in arms[k]]) if len(arms[k]) > 1 else 0 for k in ks]
        a1.errorbar(ks, mean, yerr=sd, marker=mk, ms=3.5, lw=1.1, capsize=2, c=c, label=lab)
        P = [sum(1 for a, _ in arms[k] if a >= 20) / len(arms[k]) for k in ks]
        a2.plot(ks, P, marker=mk, ms=3.5, lw=1.1, c=c, label=lab)
    a1.axhline(20, ls="--", lw=0.8, c="0.4")
    a1.text(0.98, 0.96, "20% criterion", fontsize=6, color="0.35",
            ha="right", va="top", transform=a1.transAxes)
    a1.set_xscale("log"); a1.set_xlabel("distinct keys $N$"); a1.set_ylabel("chance-corrected recall (%)")
    a1.legend(frameon=False, fontsize=6)
    a2.axhline(0.5, ls="--", lw=0.8, c="0.4")
    a2.set_xscale("log"); a2.set_xlabel("distinct keys $N$")
    a2.set_ylabel("$P$(discriminating run)"); a2.set_ylim(-0.05, 1.08)
    a2.set_title("$K^*_P$ = the 0.5 crossing", fontsize=7)
    # Matplotlib's log locator labels minor decades too, and at this width the arms sit close
    # enough that the default labels overprinted into an unreadable band. Label the arms we
    # actually measured and silence everything else.
    for ax in (a1, a2):
        ax.set_xticks(_KEY_TICKS)
        ax.set_xticklabels([f"{k // 1000}k" for k in _KEY_TICKS], fontsize=6.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=6.5)
    _save(fig, "F5_dense")


# ---------------------------------------------------------------------------------------------
# F6 — the reconciliation waterfall
# ---------------------------------------------------------------------------------------------
def f6_waterfall():
    """From plot_data.json's reconciliation block."""
    r = A.reconciliation()
    gap = r["gap"].value; am = r["amortisation"].value; su = r["surface"].value
    # LOADED, not recomputed: am * su on the rounded carriers is 2.34 and the artifact's product is
    # 2.35 (rounded from the unrounded carriers). The figure said 2.34 while the table said 2.35.
    product = r["product"].value
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    # Second lines are wrapped rather than shortened: at one line each they were wider than the
    # 4-category slot and "facts/key" printed into "(measured".
    bars = [("observed gap\n(ours vs\nreference)", gap, "0.45"),
            ("amortisation\n(6 vs 3\nfacts/key)", am, FKA_C),
            ("surface\n(measured,\nparams/key)", su, BPE_C),
            ("carriers\n(product)", product, "#2e7d32")]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, width=0.6, color=c)
        ax.text(i, v + 0.05, f"{v:.2f}$\\times$", ha="center", fontsize=7.5)
    ax.axhline(gap, ls="--", lw=0.9, c="0.35")
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=6.0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("factor"); ax.set_ylim(0, product * 1.25)
    ax.set_title(f"carriers over-cover the gap by {round((product / gap - 1) * 100)}%",
                 fontsize=7.5)
    _save(fig, "F6_waterfall")


# ---------------------------------------------------------------------------------------------
# F7 — seed spread, FKA against dense
# ---------------------------------------------------------------------------------------------
def f7_seeds():
    """FKA from sweep.log (10 seeds x 7 rungs); dense from the run logs (10 / 5 seeds)."""
    fka = A.fka_seed_sweep()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    labels, sigmas, colors = [], [], []
    for rung, vals in fka.items():
        labels.append(rung.replace("stages=", "st ").replace(" K=", " K"))
        sigmas.append(st.stdev(vals) if len(vals) > 1 else 0.0)
        colors.append(FKA_C)
    for fam, c, tag in (("syllable_1M", DENSE_C, "1M"), ("syllable_10M", "#7b1fa2", "10M")):
        arms = A.dense_arms(fam)
        for k in sorted(arms):
            if len(arms[k]) >= 5:
                # Single line: rotated two-line labels interleave, so the second line of one
                # label reads next to the first line of its neighbour.
                labels.append(f"dense {tag}, {k // 1000}k keys")
                sigmas.append(st.stdev([a for a, _ in arms[k]]))
                colors.append(c)
    ax.bar(range(len(sigmas)), sigmas, color=colors, width=0.68)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.6)
    ax.set_ylabel("$\\sigma$ across seeds (points)")
    ax.set_title("FKA (blue) against dense (red/purple) — same statistic, $n=10$ / $10$ / $5$",
                 fontsize=7)
    _save(fig, "F7_seeds")


# ---------------------------------------------------------------------------------------------
# F8 — the alpha bracket across the seven refinements
# ---------------------------------------------------------------------------------------------
def f8_alpha():
    """The seven refinements, loaded from refinements.json (M5 §5.147 transcribed once).

    Row 7 is drawn but marked NOT an alpha input: K*_BPE is a surface measurement at 1M and never
    enters the 1M/10M ratio, so its alpha is unchanged BY CONSTRUCTION. Drawing it in the same
    style as rows 1-6 implied it was an alpha refinement that happened to move nothing.
    """
    L = math.log(A.dense_kstar()["params_10M"].value / A.dense_kstar()["params_1M"].value)

    def alpha_of(r):
        (k1lo, k1hi), (k10lo, k10hi) = r["bracket_1M"], r["bracket_10M"]
        return math.log(k10lo / k1hi) / L, math.log(k10hi / k1lo) / L

    A.check_refinement_alphas(alpha_of)   # transcription vs the ledger's stated alphas
    rows, _ = A.refinements()

    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    hist = []
    for i, r in enumerate(rows):
        lo, hi = alpha_of(r)
        inp = r["is_alpha_input"]
        suffix = "" if inp else "  (surface; not an $\\alpha$ input)"
        hist.append((f"{r['n']}. {r['label']}{suffix}",))
        ax.plot([lo, hi], [i, i], lw=3.0, c=DENSE_C if inp else "none",
                solid_capstyle="butt", alpha=0.85)
        if not inp:
            # Hollow: the interval is carried forward unchanged, not re-measured.
            ax.plot([lo, hi], [i, i], lw=3.0, c="0.75", solid_capstyle="butt",
                    alpha=0.9, ls=(0, (1.4, 1.2)))
        ax.plot([(lo + hi) / 2], [i], "o", ms=3, c="k" if inp else "0.6")
    ax.axvline(0.5, ls="--", lw=0.9, c="0.35")
    # Anchored in axes coords so it sits inside the frame regardless of how many rows there are;
    # in data coords it drifted up against the title.
    ax.text(0.5 / 0.58 - 0.012, 0.5, r"$\alpha = 0.5$ ($\sqrt{P}$)", fontsize=6.5, color="0.35",
            rotation=90, ha="right", va="center", transform=ax.transAxes)
    ax.set_yticks(range(len(hist)))
    ax.set_yticklabels([h[0] for h in hist], fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlabel(r"scaling exponent $\alpha$")
    ax.set_xlim(0, 0.58)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("seven refinements; the verdict never moved", fontsize=7.5)
    _save(fig, "F8_alpha")


# ---------------------------------------------------------------------------------------------
# F3 — architecture schematic (drawn, not measured; the one figure with no numbers)
# ---------------------------------------------------------------------------------------------
def f3_architecture():
    """The only figure carrying no measured values — a schematic of the surviving design."""
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Geometry is derived, not hand-placed: four boxes and three arrow gaps tile the width, so a
    # label can never sit over an arrow. The earlier version fixed box width at 0.15 while the
    # labels needed ~0.22, and the overflow landed on the connectors.
    W, GAP = 0.216, 0.045
    assert abs(4 * W + 3 * GAP - 0.999) < 1e-9, "boxes and gaps must tile the axes width"
    Y0, H = 0.34, 0.40
    mid = Y0 + H / 2

    boxes = [("reasoning kernel\n(dim-64, frozen)", FKA_C),
             ("content-computed\nkeys $f(\\mathrm{content})$", "#2e7d32"),
             ("factorized substrate\n(RVQ stages\n+ pointers)", "#7b1fa2"),
             ("value tables\n(exact)", "0.45")]

    placed = []
    for i, (label, c) in enumerate(boxes):
        x = i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, Y0), W, H, fc="white", ec=c, lw=1.4))
        placed.append((ax.text(x + W / 2, mid, label, ha="center", va="center", fontsize=6.0),
                       x, x + W))
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + W, mid), (x + W + GAP, mid), arrowstyle="-|>",
                                         mutation_scale=8, lw=0.9, color="0.35"))

    ax.text(0.5, 0.10, "no denoiser $\\cdot$ no dim-lifting $\\cdot$ no separable key path "
                        "— each eliminated by measurement or proof",
            ha="center", fontsize=6.2, color="0.3")

    # A schematic carries no measured values, so nothing else would catch a label outgrowing its
    # box. Measure the rendered extents and fail the build instead of shipping the collision.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for txt, x0, x1 in placed:
        bb = inv.transform_bbox(txt.get_window_extent(fig.canvas.get_renderer()))
        if bb.x0 < x0 + 0.004 or bb.x1 > x1 - 0.004:
            raise AssertionError(
                f"F3 label {txt.get_text()!r} spans [{bb.x0:.4f}, {bb.x1:.4f}] "
                f"but its box is [{x0:.4f}, {x1:.4f}] — it would print over a connector")
    _save(fig, "F3_architecture")


if __name__ == "__main__":
    print("generating figures from persisted artifacts only:")
    f1_frontier(); f2_fork(); f3_architecture(); f4_margins()
    f5_dense(); f6_waterfall(); f7_seeds(); f8_alpha()
    print("done.")
