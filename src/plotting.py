"""Figure generation. One function per figure type.

Every function reads previously computed data — batch aggregates from
results/raw/runs.csv, curves from the per-cell metric arrays — nothing
is recomputed here. Figures for the report land in results/figures/ only
when produced from real metric runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams.update({
    "savefig.dpi": 150,
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})

SYSTEM_COLORS = {
    "none": "#888888",
    "nlms_f64": "#1f77b4",
    "speex": "#ff7f0e",
    "nlms_q15": "#d62728",
}

# Reader-facing names for internal identifiers; data/CSV keys stay internal.
SYSTEM_LABELS = {
    "none": "Passthrough",
    "nlms_f64": "Float NLMS",
    "nlms_q15": "Q15 NLMS",
    "speex": "SpeexDSP",
}
LEVEL_LABELS = {
    "no_noise": "No noise", "snr20": "20 dB SNR", "snr10": "10 dB SNR",
    "far_single": "Far-end only", "double": "Double-talk",
    "near_single": "Near-end only", "double_talk": "Double-talk",
    "50ms": "50 ms", "100ms": "100 ms", "200ms": "200 ms",
    "400ms": "400 ms",
    "mu0.1": "0.1", "mu0.3": "0.3", "mu0.5": "0.5", "mu0.9": "0.9",
}



def plot_segmentation_diagnostic(run_dir: Path, out_path: Path) -> None:
    """Component envelopes plus the resulting three-state strip."""
    z = np.load(run_dir / "signals.npz")
    seg = np.load(run_dir / "segmentation.npz")
    fs = int(z["sample_rate"])
    frame_len = int(seg["frame_len"])
    far, near = seg["far_active"], seg["near_active"]
    n_frames = len(far)
    t_frames = (np.arange(n_frames) + 0.5) * frame_len / fs

    def envelope_db(sig):
        n = len(sig) // frame_len
        p = np.mean(sig[: n * frame_len].reshape(n, frame_len) ** 2, axis=1)
        return 10 * np.log10(np.maximum(p, 1e-12))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t_frames, envelope_db(z["d_echo"]), label="d_echo (echo at mic)",
             lw=0.9)
    ax1.plot(t_frames, envelope_db(z["s"]), label="s (near-end at mic)",
             lw=0.9)
    ax1.set_ylabel("frame power (dBov)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # State strip: 0 neither, 1 far only, 2 double talk, 3 near only.
    state = np.zeros(n_frames)
    state[far & ~near] = 1
    state[far & near] = 2
    state[~far & near] = 3
    ax2.imshow(state[np.newaxis, :], aspect="auto", interpolation="nearest",
               extent=(0, t_frames[-1] + frame_len / fs / 2, 0, 1),
               cmap=matplotlib.colors.ListedColormap(
                   ["#dddddd", "#1f77b4", "#9467bd", "#2ca02c"]),
               vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_xlabel("time (s)")
    ax2.set_title("state: grey=neither, blue=far only (ERLE-valid), "
                  "purple=double talk, green=near only", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_curves(csv_path: Path, cell_dir: Path, out_dir: Path,
                         systems: list[str] = ("none", "nlms_f64", "speex",
                                               "nlms_q15"),
                         seed: int = 0) -> None:
    """Baseline-cell ERLE curves (plus the recorded NLMS misalignment
    trajectory) for one seed. Curves come from the persisted per-system
    metric arrays; steady-state reference lines come from runs.csv."""
    df = pd.read_csv(csv_path)
    rows = df[(df["stage"] == "a") & (df["seed"] == seed)
              & (df["scenario_key"].str.startswith("rt0.4_d1_"))]

    fig, ax = plt.subplots(figsize=(8.7, 4.25), layout="constrained")
    for system in systems:
        m = np.load(cell_dir / f"metrics_{system}.npz")
        color = SYSTEM_COLORS.get(system)
        ax.plot(m["frame_times_s"], m["erle_smoothed_db"], lw=1.2,
                label=SYSTEM_LABELS.get(system, system), color=color)
        ss = rows.loc[rows["system"] == system, "erle_steady_state_db"]
        if len(ss) and np.isfinite(ss.iloc[0]):
            ax.axhline(ss.iloc[0], ls=":", lw=0.8, color=color, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("smoothed ERLE (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"ERLE — baseline cell, seed {seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "baseline_erle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 3.8), layout="constrained")
    plotted = False
    for system in ("nlms_f64", "nlms_q15"):
        path = cell_dir / f"metrics_{system}.npz"
        if not path.exists():
            continue
        m = np.load(path)
        if "misalignment_curve_db" not in m:
            continue
        ax.plot(m["misalignment_times_s"], m["misalignment_curve_db"],
                lw=1.2, color=SYSTEM_COLORS[system],
                label=SYSTEM_LABELS.get(system, system))
        plotted = True
    if plotted:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("misalignment (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Misalignment — baseline cell")
        fig.savefig(out_dir / "baseline_misalignment.png", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch figures — every function reads only results/raw/runs.csv
# ---------------------------------------------------------------------------

def _rt60_row_labels(sub: pd.DataFrame) -> list[str]:
    """Row labels carrying achieved RT60 (range across distances)."""
    labels = []
    for rt in sorted(sub["rt60_target_s"].unique()):
        ach = sub.loc[sub["rt60_target_s"] == rt, "rt60_achieved_echo_s"]
        labels.append(f"{rt:g}\n(ach {ach.min():.2f}–{ach.max():.2f})")
    return labels


def _stage_a_heatmap(df: pd.DataFrame, value_col: str, label: str,
                     out_path: Path, fmt: str = "{:.1f}") -> None:
    sub = df[(df["stage"] == "a") & (df["system"] != "none")]
    systems = sorted(sub["system"].unique())
    rts = sorted(sub["rt60_target_s"].unique())
    dists = sorted(sub["speaker_mic_distance_m"].unique())

    fig, axes = plt.subplots(1, len(systems),
                             figsize=(3.5 * len(systems), 3.9), sharey=True,
                             layout="constrained")
    vmin = sub[value_col].min()
    vmax = sub[value_col].max()
    for ax, system in zip(np.atleast_1d(axes), systems):
        grid = np.full((len(rts), len(dists)), np.nan)
        for i, rt in enumerate(rts):
            for j, dist in enumerate(dists):
                cell = sub[(sub["system"] == system)
                           & (sub["rt60_target_s"] == rt)
                           & (sub["speaker_mic_distance_m"] == dist)]
                grid[i, j] = cell[value_col].mean()
        im = ax.imshow(grid, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        for i in range(len(rts)):
            for j in range(len(dists)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, fmt.format(grid[i, j]), ha="center",
                            va="center", color="white", fontsize=12,
                            path_effects=[
                                matplotlib.patheffects.withStroke(
                                    linewidth=2, foreground="black")])
        ax.set_xticks(range(len(dists)), [f"{d:g}" for d in dists])
        ax.set_yticks(range(len(rts)), _rt60_row_labels(sub))
        ax.set_xlabel("speaker–mic distance (m)")
        ax.set_title(SYSTEM_LABELS.get(system, system))
    np.atleast_1d(axes)[0].set_ylabel("RT60 target (s), achieved range")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label=label, shrink=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_a(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    _stage_a_heatmap(df, "erle_steady_state_db", "steady-state ERLE (dB)",
                     out_dir / "stage_a_erle.png")
    _stage_a_heatmap(df, "convergence_time_s", "convergence time (s)",
                     out_dir / "stage_a_convergence.png", fmt="{:.2f}")


def _strip_axis(ax, sub: pd.DataFrame, levels: list, value_col: str,
                level_col: str = "level") -> None:
    """Per-seed points, jittered per system; diverged rows drawn as x."""
    offsets = {"none": -0.3, "nlms_f64": -0.1, "nlms_q15": 0.1,
               "speex": 0.3}
    for system, grp in sub.groupby("system"):
        color = SYSTEM_COLORS.get(system, "#333333")
        sys_label = SYSTEM_LABELS.get(system, system)
        for k, level in enumerate(levels):
            pts = grp[grp[level_col] == level]
            if pts.empty:
                continue
            xs = k + offsets.get(system, 0.0) + \
                (pts["seed"].to_numpy() - 1) * 0.045
            ys = pts[value_col].to_numpy(dtype=float)
            ok = pts["status"].to_numpy() == "ok"
            ax.scatter(xs[ok], ys[ok], s=28, color=color, zorder=3,
                       label=sys_label if k == 0 else None)
            ax.scatter(xs[~ok], ys[~ok], s=44, color=color, marker="x",
                       zorder=3,
                       label=f"{sys_label} (diverged)" if (~ok).any()
                       and k >= 0
                       and f"{sys_label} (diverged)" not in
                       [h.get_label() for h in ax.collections] else None)
            finite = ys[np.isfinite(ys)]
            if len(finite):
                ax.hlines(np.mean(finite), k - 0.3, k + 0.3, color=color,
                          lw=1.0, alpha=0.5)
    ax.set_xticks(range(len(levels)),
                  [LEVEL_LABELS.get(str(lv), str(lv)) for lv in levels])
    ax.grid(alpha=0.3, axis="y")


def plot_stage_b_talk(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "talk")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.7, 4.2),
                                   layout="constrained")
    # ERLE is undefined with no far-end activity, so near_single is not
    # on this axis at all (no placeholder slot), mirroring the right
    # panel's rule for STOI under far_single.
    _strip_axis(ax1, sub, ["far_single", "double"], "erle_steady_state_db")
    ax1.set_ylabel("steady-state ERLE (dB)")
    ax1.set_title("echo suppression (far-end-only regions)")
    ax1.legend()
    # STOI exists only where the near end is active; far_single is
    # intentionally absent from this panel.
    _strip_axis(ax2, sub, ["double", "near_single"], "stoi")
    ax2.set_ylabel("STOI")
    ax2.set_title("near-end intelligibility (near-active regions)")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_talk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_noise(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "noise")]
    levels = ["no_noise", "snr20", "snr10"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1), layout="constrained")
    _strip_axis(ax, sub, levels, "erle_steady_state_db")
    ax.set_ylabel("steady-state ERLE (dB)")
    ax.set_xlabel("background noise")
    ax.legend()
    ax.set_title("ERLE under background noise")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_tail(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "tail_length")]
    levels = ["50ms", "100ms", "200ms", "400ms"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1), layout="constrained")
    _strip_axis(ax, sub, levels, "erle_steady_state_db")
    ax.set_ylabel("steady-state ERLE (dB)")
    ax.set_xlabel("filter length")
    ax.legend()
    ax.set_title("ERLE vs filter length")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_tail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_mu(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "mu")]
    swept = sub[sub["level"] != "reference"]
    ref = sub[sub["level"] == "reference"]
    levels = sorted(swept["level"].unique(), key=lambda s: float(s[2:]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.7, 4.2),
                                   layout="constrained")
    # SpeexDSP exposes no coefficients, so it has no misalignment and
    # appears only on the ERLE panel (as the unswept reference line).
    for ax, col, ylabel in [(ax1, "erle_steady_state_db",
                             "steady-state ERLE (dB)"),
                            (ax2, "misalignment_final_db",
                             "final misalignment (dB)")]:
        _strip_axis(ax, swept, levels, col)
        if col == "erle_steady_state_db" and not ref.empty:
            ax.axhline(ref[col].mean(), color=SYSTEM_COLORS["speex"],
                       ls="--", lw=1.2,
                       label="SpeexDSP @ baseline (reference, not swept)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("NLMS step size μ")
        ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_mu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_word_length(csv_path: Path, out_dir: Path) -> None:
    """Headline figure: effective coefficient word length vs steady-state
    ERLE and convergence time, per-seed points, with the float NLMS at
    the same scenario as a reference line and 0 dB (= no processing)
    marked. Non-converged runs are drawn at the axis top as open
    triangles rather than silently dropped."""
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "word_length")]
    levels = ["15bit", "11bit", "9bit", "7bit"]
    f64_ref = df[(df["stage"] == "a") & (df["system"] == "nlms_f64")
                 & (df["scenario_key"].str.startswith("rt0.4_d1_"))]
    color = SYSTEM_COLORS["nlms_q15"]
    ref_color = SYSTEM_COLORS["nlms_f64"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.7, 4.35),
                                   layout="constrained")
    for ax, col, ylabel in [(ax1, "erle_steady_state_db",
                             "steady-state ERLE (dB)"),
                            (ax2, "misalignment_final_db",
                             "final misalignment (dB)")]:
        for k, level in enumerate(levels):
            pts = sub[sub["level"] == level]
            xs = k + (pts["seed"].to_numpy() - 1) * 0.06
            ys = pts[col].to_numpy(dtype=float)
            ok = np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], s=34, color=color, zorder=3,
                       label="Q15 NLMS (masked)" if k == 0 else None)
            if len(ys[ok]):
                ax.hlines(np.mean(ys[ok]), k - 0.28, k + 0.28, color=color,
                          lw=1.1, alpha=0.6)
            if (~ok).any():  # non-converged: flag at top, never dropped
                top = np.nanmax(sub[col].to_numpy(dtype=float))
                ax.scatter(xs[~ok], np.full((~ok).sum(), top), s=44,
                           facecolors="none", edgecolors=color, marker="^",
                           zorder=3,
                           label="not converged" if
                           "not converged" not in
                           [h.get_label() for h in ax.collections]
                           else None)
        ax.axhline(f64_ref[col].mean(), color=ref_color, ls="--", lw=1.2,
                   label="Float NLMS, same scenario (reference)")
        if col == "erle_steady_state_db":
            ax.axhline(0.0, color="#888888", ls=":", lw=1.0,
                       label="0 dB (no processing)")
        ax.set_xticks(range(len(levels)),
                      [lv.replace("bit", "") for lv in levels])
        ax.set_xlabel("effective coefficient word length (bits)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "word_length_sweep.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_q15_float_divergence(cell_dir: Path, out_dir: Path,
                              sample_rate: int = 16000) -> None:
    """Baseline cell: per-sample coefficient divergence between the Q15
    filter and its in-loop float64 shadow on identical quantised input."""
    path = cell_dir / "e_nlms_q15.npz"
    if not path.exists():
        return
    z = np.load(path)
    div = z["coeff_div_db"]
    dec = 16  # plot decimation only; the stored curve is per-sample
    t = np.arange(len(div))[::dec] / sample_rate
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(t, div[::dec], lw=0.9, color=SYSTEM_COLORS["nlms_q15"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("10·log10(‖w_q15/2¹⁵ − w_float‖² / ‖w_float‖²)  (dB)")
    ax.grid(alpha=0.3)
    ax.set_title("Q15 divergence from float shadow")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "q15_float_divergence.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_audibility(csv_path: Path, out_dir: Path) -> None:
    """Residual-echo audibility at the baseline cell (RT60 0.4 s, 1.0 m,
    far-end only, no noise), per system: fraction of TF units above the
    masking threshold and mean excess over it. In this cell the speex
    residual from the two-run isolation is exact: d == d_echo, so the
    echo-only rerun is bit-identical to the primary run."""
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "a") & (df["level"] == "rt0.4_d1")]
    systems = ["none", "nlms_f64", "nlms_q15", "speex"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2),
                                   layout="constrained")
    for ax, col, ylabel in [(ax1, "audibility_fraction",
                             "fraction of TF units above threshold"),
                            (ax2, "audibility_excess_db",
                             "mean excess above threshold (dB)")]:
        for k, system in enumerate(systems):
            pts = sub[sub["system"] == system]
            xs = k + (pts["seed"].to_numpy() - 1) * 0.08
            ys = pts[col].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=34, color=SYSTEM_COLORS[system], zorder=3)
            ax.hlines(np.nanmean(ys), k - 0.25, k + 0.25,
                      color=SYSTEM_COLORS[system], lw=1.1, alpha=0.6)
        ax.set_xticks(range(len(systems)),
                      [SYSTEM_LABELS[s] for s in systems])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title("audibility — baseline cell")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "audibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def batch_figures(csv_path: Path, out_dir: Path,
                  baseline_cell_dir: Path | None = None) -> None:
    plot_stage_a(csv_path, out_dir)
    plot_stage_b_talk(csv_path, out_dir)
    plot_stage_b_noise(csv_path, out_dir)
    plot_stage_b_tail(csv_path, out_dir)
    plot_stage_b_mu(csv_path, out_dir)
    plot_word_length(csv_path, out_dir)
    plot_audibility(csv_path, out_dir)
    if baseline_cell_dir is not None and baseline_cell_dir.is_dir():
        plot_baseline_curves(csv_path, baseline_cell_dir, out_dir)
        plot_q15_float_divergence(baseline_cell_dir, out_dir)
