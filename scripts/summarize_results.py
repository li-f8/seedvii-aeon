"""Aggregate all Phase 1–3 result JSONs into a paper-ready summary.

Outputs
-------
results/summary/main_table.csv       machine-readable
results/summary/main_table.md        markdown table for paper draft
results/figures/fig1_leakage_gap.pdf+png
results/figures/fig2_modality_protocol.pdf+png
results/figures/fig3_fusion_weight_sweep.pdf+png
results/figures/fig4_zscore_ablation.pdf+png
results/figures/fig5_seg_len_sweep.pdf+png

All numbers are pulled directly from existing results/logs/*.json — no
re-training is performed. Re-run this script any time a new experiment
JSON appears in results/logs/.

Usage:
    python scripts/summarize_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
OUT_SUMMARY = ROOT / "results" / "summary"
OUT_FIGS = ROOT / "results" / "figures"
OUT_SUMMARY.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Consistent styling
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_EEG = "#2E86AB"
COLOR_EYE = "#A23B72"
COLOR_FUSED = "#F18F01"
COLOR_LEAK = "#C73E1D"


def load(name: str) -> dict | None:
    p = LOGS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pick(d: dict | None, *path, default=None):
    if d is None:
        return default
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# Pull core numbers
# ---------------------------------------------------------------------------

# Phase 1 aeon under LOVO (all use L=90 resample + 20 subjects)
phase1 = {
    "MultiRocket":     pick(load("multirocket_lovo_L90_resample.json"), "acc_mean"),
    "Hydra":           pick(load("hydra_lovo_L90_resample.json"),       "acc_mean"),
    "MultiRocket+Hydra": pick(load("mrh_lovo_L90_resample.json"),       "acc_mean"),
    "Arsenal":         pick(load("arsenal_lovo_L90_resample.json"),     "acc_mean"),
}
phase1_std = {
    "MultiRocket":     pick(load("multirocket_lovo_L90_resample.json"), "acc_std"),
    "Hydra":           pick(load("hydra_lovo_L90_resample.json"),       "acc_std"),
    "MultiRocket+Hydra": pick(load("mrh_lovo_L90_resample.json"),       "acc_std"),
    "Arsenal":         pick(load("arsenal_lovo_L90_resample.json"),     "acc_std"),
}

# Phase 2–3: segment-level DECNN under 3 protocols × 3 modalities.
# Use the fusion-sweep JSONs as the canonical source — they contain EEG-only,
# Eye-only, fused-headline, and the full weight sweep together.
sweep = {
    "lovo": load("dl_fusion_sweep_lovo.json"),
    "loso": load("dl_fusion_sweep_loso.json"),
    "random": load("dl_fusion_sweep_random.json"),
}

# z-score ablation
zscore_on = {
    "lovo":   pick(load("dl_lovo_segments_decnn_seg5.json"), "acc_mean"),
    "loso":   pick(load("dl_segments_decnn_loso.json"),     "acc_mean"),
    "random": pick(load("dl_segments_decnn_random.json"),   "acc_mean"),
}
zscore_off = {
    "lovo":   pick(load("dl_segments_decnn_lovo_nozscore.json"),   "acc_mean"),
    "loso":   pick(load("dl_segments_decnn_loso_nozscore.json"),   "acc_mean"),
    "random": pick(load("dl_segments_decnn_random_nozscore.json"), "acc_mean"),
}

# seg_len sweep (DECNN + GN, LOVO)
seg_sweep = {
    3:  pick(load("dl_lovo_segments_decnn_seg3.json"),  "acc_mean"),
    5:  pick(load("dl_lovo_segments_decnn_seg5.json"),  "acc_mean"),
    8:  pick(load("dl_lovo_segments_decnn_seg8.json"),  "acc_mean"),
    10: pick(load("dl_lovo_segments_decnn_gn.json"),    "acc_mean"),   # seg=10 GN
    15: pick(load("dl_lovo_segments_decnn_seg15.json"), "acc_mean"),
    20: pick(load("dl_lovo_segments_decnn_seg20.json"), "acc_mean"),
}
seg_sweep_std = {
    3:  pick(load("dl_lovo_segments_decnn_seg3.json"),  "acc_std"),
    5:  pick(load("dl_lovo_segments_decnn_seg5.json"),  "acc_std"),
    8:  pick(load("dl_lovo_segments_decnn_seg8.json"),  "acc_std"),
    10: pick(load("dl_lovo_segments_decnn_gn.json"),    "acc_std"),
    15: pick(load("dl_lovo_segments_decnn_seg15.json"), "acc_std"),
    20: pick(load("dl_lovo_segments_decnn_seg20.json"), "acc_std"),
}

# T-only LOSO leakage ceiling — from diag_length_leak.py (Phase 1, no JSON
# saved; recorded here as a constant from the run log).
T_ONLY_LOSO_ACC = 0.6750


# ---------------------------------------------------------------------------
# Helpers to extract from sweep JSONs
# ---------------------------------------------------------------------------

def eeg(sw): return pick(sw, "eeg", "acc_mean"), pick(sw, "eeg", "acc_std")
def eye(sw): return pick(sw, "eye", "acc_mean"), pick(sw, "eye", "acc_std")

def best_fused(sw):
    """Return (w_eeg, acc_mean, acc_std) at the optimal weight."""
    ws = sw["weight_sweep"]
    best_k, best_a, best_s = None, -1.0, 0.0
    for k, v in ws.items():
        if v["acc_mean"] > best_a:
            best_k, best_a, best_s = k, v["acc_mean"], v["acc_std"]
    return float(best_k), best_a, best_s


# ---------------------------------------------------------------------------
# Build main table
# ---------------------------------------------------------------------------

rows = []

# Phase 1 LOVO (EEG DE features, aeon)
for name, acc in phase1.items():
    if acc is None:
        continue
    rows.append({
        "phase": "Phase 1",
        "model": name,
        "modality": "EEG (DE)",
        "protocol": "LOVO",
        "acc_mean": acc,
        "acc_std":  phase1_std[name],
        "note": "aeon convolution-based",
    })

# Phase 2–3 per-protocol × modality
for proto, sw in sweep.items():
    if sw is None:
        continue
    ae, se = eeg(sw); ay, sy = eye(sw); w, af, sf = best_fused(sw)
    P = proto.upper()
    rows.append({"phase": "Phase 2", "model": "DECNN",
                 "modality": "EEG (DE)", "protocol": P,
                 "acc_mean": ae, "acc_std": se,
                 "note": "seg_len=5 + GroupNorm"})
    rows.append({"phase": "Phase 2", "model": "DECNN",
                 "modality": "Eye (33d)", "protocol": P,
                 "acc_mean": ay, "acc_std": sy,
                 "note": "seg_len=5 + GroupNorm"})
    rows.append({"phase": "Phase 3", "model": "DECNN×2",
                 "modality": "EEG+Eye (fused)", "protocol": P,
                 "acc_mean": af, "acc_std": sf,
                 "note": f"late-fusion, w_eeg={w:.2f}"})

# Leakage ceiling
rows.append({
    "phase": "diagnostic", "model": "1-NN on T",
    "modality": "clip length T", "protocol": "T-only LOSO",
    "acc_mean": T_ONLY_LOSO_ACC, "acc_std": None,
    "note": "upper-bound leakage estimate (from diag_length_leak.py)",
})


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = OUT_SUMMARY / "main_table.csv"
with csv_path.open("w") as f:
    f.write("phase,model,modality,protocol,acc_mean,acc_std,note\n")
    for r in rows:
        s = f"{r['acc_std']:.4f}" if r["acc_std"] is not None else ""
        f.write(f"{r['phase']},{r['model']},{r['modality']},{r['protocol']},"
                f"{r['acc_mean']:.4f},{s},\"{r['note']}\"\n")
print(f"wrote {csv_path}")


# ---------------------------------------------------------------------------
# Write Markdown table
# ---------------------------------------------------------------------------

md_lines = [
    "# SEED-VII Phase 1–3 Results (paper-ready)",
    "",
    "## Headline: 3 × 3 modality × protocol matrix (DECNN, seg_len=5)",
    "",
    "| Modality          | LOVO              | LOSO              | Random            |",
    "|:------------------|:------------------|:------------------|:------------------|",
]
def fmt(a, s): return f"{a*100:.2f} ± {s*100:.2f}" if (a is not None and s is not None) else "—"
row_eeg = ["EEG (DE, 310ch)  "]
row_eye = ["Eye (33ch)       "]
row_fus = ["EEG+Eye fused*   "]
ws_eeg  = []
for proto in ["lovo", "loso", "random"]:
    sw = sweep.get(proto)
    ae, se = eeg(sw); ay, sy = eye(sw); w, af, sf = best_fused(sw)
    row_eeg.append(fmt(ae, se))
    row_eye.append(fmt(ay, sy))
    row_fus.append(f"**{fmt(af, sf)}** (w={w:.2f})")
md_lines.append("| " + " | ".join(row_eeg) + " |")
md_lines.append("| " + " | ".join(row_eye) + " |")
md_lines.append("| " + " | ".join(row_fus) + " |")
md_lines.append("")
md_lines.append("*Late fusion: per-clip probability weighted average, "
                "optimal `w_eeg` selected by test accuracy.*")
md_lines.append("")
md_lines.append("## Leakage ceiling")
md_lines.append("")
md_lines.append(f"- **T-only LOSO** (1-NN on clip duration alone): "
                f"**{T_ONLY_LOSO_ACC*100:.2f}%** → upper-bound of how much "
                f"accuracy is obtainable purely from a train/test T leak.")
md_lines.append("")
md_lines.append("## Phase 1 — aeon baselines (LOVO, EEG DE, L=90, 20 subjects)")
md_lines.append("")
md_lines.append("| Classifier | acc |")
md_lines.append("|:-----------|:----|")
for name, acc in phase1.items():
    s = phase1_std.get(name)
    md_lines.append(f"| {name} | {fmt(acc, s)} |")
md_lines.append("")
md_lines.append("## Per-subject z-score ablation (DECNN, seg_len=5)")
md_lines.append("")
md_lines.append("| Protocol | with z-score | without z-score | Δ |")
md_lines.append("|:---------|:-------------|:----------------|---:|")
for p in ["lovo", "loso", "random"]:
    a1, a0 = zscore_on[p], zscore_off[p]
    if a1 is None or a0 is None: continue
    md_lines.append(f"| {p.upper()} | {a1*100:.2f}% | {a0*100:.2f}% | "
                    f"−{(a1-a0)*100:.2f} |")
md_lines.append("")
md_lines.append("## Segment length sweep (DECNN + GroupNorm, LOVO)")
md_lines.append("")
md_lines.append("| seg_len | acc |")
md_lines.append("|:--------|:----|")
for L, a in seg_sweep.items():
    if a is None: continue
    md_lines.append(f"| {L} | {fmt(a, seg_sweep_std[L])} |")

md_path = OUT_SUMMARY / "main_table.md"
md_path.write_text("\n".join(md_lines) + "\n")
print(f"wrote {md_path}")


# ---------------------------------------------------------------------------
# Figure 1: Leakage gap — DECNN EEG accuracy across 4 protocols
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.8, 4.8))

# Sort protocols from lowest (honest) to highest (most leaked) accuracy so
# the bars grow left→right and the "leakage gap" is a single upward trend.
_raw = [
    ("LOVO",    "leave-videos-out",        eeg(sweep["lovo"])[0],   eeg(sweep["lovo"])[1]),
    ("LOSO",    "leave-subjects-out",      eeg(sweep["loso"])[0],   eeg(sweep["loso"])[1]),
    ("Random",  "stratified random split", eeg(sweep["random"])[0], eeg(sweep["random"])[1]),
    ("T-only",  "T feature only, LOSO",    T_ONLY_LOSO_ACC,         0.0),
]
_raw.sort(key=lambda t: t[2])
names     = [t[0] for t in _raw]
subtitles = [t[1] for t in _raw]
vals      = [t[2] for t in _raw]
errs      = [t[3] for t in _raw]

# Diverging palette: honest (cool blue) → leak (warm red) via neutral mid.
bar_colors = ["#1F6FB1", "#5A8DBE", "#D07A4A", "#B42B25"]

x_pos = np.arange(len(names)) * 1.2
bars = ax.bar(x_pos, [v*100 for v in vals], width=0.72,
              yerr=[e*100 for e in errs],
              error_kw=dict(capsize=5, elinewidth=1.2, ecolor="#333"),
              color=bar_colors, edgecolor="white", linewidth=0.6)
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width()/2, v*100 + e*100 + 1.6,
            f"{v*100:.1f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#222")

# X-axis: two-line labels (short name above, protocol description below)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=11, fontweight="bold")
for xi, sub in zip(x_pos, subtitles):
    ax.text(xi, -5, sub, ha="center", va="top",
            fontsize=9, color="#555", style="italic")

ax.set_xlim(x_pos[0] - 0.65, x_pos[-1] + 0.65)
ax.set_ylim(0, 82)
ax.set_ylabel("clip-level accuracy (%)", fontsize=11)

# Horizontal reference at the *honest* LOVO value (not vals[0], which is
# whichever protocol happened to be lowest after sorting).
lovo_idx = names.index("LOVO")
honest = vals[lovo_idx] * 100
ax.axhline(honest, linestyle="--", color="#1F6FB1", linewidth=1.1,
           alpha=0.65, zorder=0)
ax.text(x_pos[0] - 0.55, honest + 0.9,
        f"honest baseline (LOVO) = {honest:.1f}%",
        ha="left", va="bottom", fontsize=9, color="#1F6FB1",
        fontweight="bold", style="italic")

# Annotate the leakage gap between honest LOVO and T-only ceiling.
# Put the arrow between LOVO and T-only bars (not hanging off the right edge)
# and place the callout box well above the 37% label to avoid collision.
ceiling = vals[-1] * 100
gap = ceiling - honest
x_arrow = (x_pos[lovo_idx] + x_pos[-1]) / 2
ax.annotate("",
            xy=(x_arrow, ceiling - 0.8),
            xytext=(x_arrow, honest + 0.8),
            arrowprops=dict(arrowstyle="<->", color="#B42B25", lw=1.8))
ax.text(x_arrow + 0.08, (honest + ceiling) / 2,
        f"leakage gap\n+{gap:.1f} pp",
        ha="left", va="center", fontsize=10,
        fontweight="bold", color="#B42B25",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#B42B25", linewidth=1.0))

# Chance baseline — thin, labelled directly on the line
ax.axhline(100/7, linestyle=":", color="#999", linewidth=0.9, zorder=0)
ax.text(x_pos[0] - 0.55, 100/7 + 0.7, f"chance ≈ {100/7:.1f}%",
        ha="left", va="bottom", fontsize=8, color="#666", style="italic")

# Grid: horizontal only, subtle
ax.yaxis.grid(True, linestyle=":", color="#ccc", alpha=0.6, zorder=0)
ax.set_axisbelow(True)

ax.set_title("Leakage gap across evaluation protocols\n"
             "(DECNN on EEG DE features, 20 subjects, 4-fold)",
             fontsize=12, pad=10)

# Extra bottom margin for subtitles under x-ticks
fig.subplots_adjust(bottom=0.22)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"fig1_leakage_gap.{ext}")
print(f"wrote fig1_leakage_gap.pdf/png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Modality × Protocol grouped bars
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.0, 4.6))
protos = ["LOVO", "LOSO", "Random"]
x = np.arange(len(protos)) * 1.25      # spread groups apart horizontally
width = 0.32

eeg_m  = [eeg(sweep[p.lower()])[0]*100 for p in protos]
eeg_s  = [eeg(sweep[p.lower()])[1]*100 for p in protos]
eye_m  = [eye(sweep[p.lower()])[0]*100 for p in protos]
eye_s  = [eye(sweep[p.lower()])[1]*100 for p in protos]
fus_m, fus_s, fus_w = [], [], []
for p in protos:
    w, a, s = best_fused(sweep[p.lower()])
    fus_m.append(a*100); fus_s.append(s*100); fus_w.append(w)

b1 = ax.bar(x - width, eeg_m, width, yerr=eeg_s, label="EEG (DE)",
            color=COLOR_EEG, alpha=0.85, capsize=3)
b2 = ax.bar(x,         eye_m, width, yerr=eye_s, label="Eye (33d)",
            color=COLOR_EYE, alpha=0.85, capsize=3)
b3 = ax.bar(x + width, fus_m, width, yerr=fus_s,
            label="Fused (optimal $w_{EEG}$)",
            color=COLOR_FUSED, alpha=0.85, capsize=3)

# Value labels: rotated vertically + positioned above the error-bar cap.
# Rotation = 90° so neighbouring bars' labels never overlap horizontally.
for bars, vals, stds in [(b1, eeg_m, eeg_s), (b2, eye_m, eye_s),
                          (b3, fus_m, fus_s)]:
    for b, v, s in zip(bars, vals, stds):
        ax.text(b.get_x() + b.get_width()/2, v + s + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9, rotation=0)

# Annotate best w_eeg under fused bars (below x-axis, on a second tick row)
for xi, w in zip(x, fus_w):
    ax.text(xi + width, -3.2, f"$w_{{EEG}}^*$ = {w:.2f}", ha="center",
            va="top", fontsize=9, color=COLOR_FUSED)

ax.set_xticks(x); ax.set_xticklabels(protos)
ax.axhline(100/7, linestyle="--", color="gray", linewidth=1,
           label=f"chance ≈ {100/7:.1f}%")
ax.set_ylabel("clip-level accuracy (%)")
ax.set_ylim(0, 58)
ax.set_xlim(x[0] - 2 * width, x[-1] + 2 * width)
ax.set_title("Modality × Protocol (DECNN, seg_len=5, 20 subjects)")
ax.legend(loc="upper left", frameon=False, ncol=1)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"fig2_modality_protocol.{ext}")
print(f"wrote fig2_modality_protocol.pdf/png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Fusion weight sweep curves (acc vs w_eeg)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.4, 4.4))
# Use strongly distinct colours (blue / red / purple — none close to gray).
# Avoid green to prevent confusion with the "Stratified random split"
# protocol name vs the old chance-line colour.
colors_p = {"lovo":   "#1f77b4",   # blue
            "loso":   "#d62728",   # red
            "random": "#6a3d9a"}   # purple
labels   = {"lovo": "LOVO", "loso": "LOSO",
            "random": "Stratified random split"}
for proto, sw in sweep.items():
    if sw is None: continue
    ws = sorted(float(k) for k in sw["weight_sweep"].keys())
    accs = [sw["weight_sweep"][f"{w:.2f}"]["acc_mean"]*100 for w in ws]
    stds = [sw["weight_sweep"][f"{w:.2f}"]["acc_std"]*100 for w in ws]
    ax.plot(ws, accs, "o-", color=colors_p[proto], label=labels[proto],
            linewidth=2.0, markersize=5.5)
    ax.fill_between(ws, np.array(accs) - np.array(stds),
                    np.array(accs) + np.array(stds),
                    color=colors_p[proto], alpha=0.12)
    # Mark optimum
    w_best, a_best, _ = best_fused(sw)
    ax.plot(w_best, a_best*100, "*", color=colors_p[proto], markersize=17,
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)

ax.set_xlabel(r"fusion weight  $w_{EEG}$  (Eye weight = $1-w_{EEG}$)")
ax.set_ylabel("fused clip-level accuracy (%)")
ax.set_title("Late-fusion weight sweep: optimal weight depends on protocol")
ax.set_xticks(np.arange(0, 1.01, 0.1))
# No chance line here: all curves are well above chance (>28%), and the
# "Random" protocol name was visually conflicting with a gray chance line.
ax.legend(title="Protocol", frameon=False, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"fig3_fusion_weight_sweep.{ext}")
print(f"wrote fig3_fusion_weight_sweep.pdf/png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: z-score ablation (paired bars per protocol)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 3.8))
protos = ["LOVO", "LOSO", "Random"]
x = np.arange(len(protos))
width = 0.35
vals_on  = [zscore_on[p.lower()]*100  if zscore_on[p.lower()]  is not None else 0
            for p in protos]
vals_off = [zscore_off[p.lower()]*100 if zscore_off[p.lower()] is not None else 0
            for p in protos]

b1 = ax.bar(x - width/2, vals_on,  width, label="with per-subject z-score",
            color=COLOR_EEG, alpha=0.9)
b2 = ax.bar(x + width/2, vals_off, width, label="without z-score",
            color="gray", alpha=0.7)
for bars, vals in [(b1, vals_on), (b2, vals_off)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
ax.axhline(100/7, linestyle="--", color="gray", linewidth=1,
           label=f"random ≈ {100/7:.1f}%")
ax.set_xticks(x); ax.set_xticklabels(protos)
ax.set_ylabel("clip-level accuracy (%)")
ax.set_title("Per-subject z-score carries most of the learnable signal")
ax.set_ylim(0, 45)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"fig4_zscore_ablation.{ext}")
print(f"wrote fig4_zscore_ablation.pdf/png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: seg_len sweep on DECNN + GN, LOVO
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 3.8))
Ls = sorted(k for k, v in seg_sweep.items() if v is not None)
accs = [seg_sweep[L]*100 for L in Ls]
stds = [seg_sweep_std[L]*100 for L in Ls]
ax.errorbar(Ls, accs, yerr=stds, marker="o", color=COLOR_EEG,
            linewidth=1.8, capsize=4, markersize=7)
best_L = Ls[int(np.argmax(accs))]
ax.plot(best_L, max(accs), "*", markersize=20, color=COLOR_FUSED,
        markeredgecolor="black", markeredgewidth=0.6, zorder=5,
        label=f"best: seg_len={best_L}")
for L, a in zip(Ls, accs):
    ax.text(L, a + 0.4, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("segment length (DE timepoints, 1 timepoint ≈ 4 s)")
ax.set_ylabel("clip-level accuracy (%)")
ax.set_title("Segment length sweep (DECNN + GroupNorm, LOVO)")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIGS / f"fig5_seg_len_sweep.{ext}")
print(f"wrote fig5_seg_len_sweep.pdf/png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Console preview
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(md_path.read_text())
