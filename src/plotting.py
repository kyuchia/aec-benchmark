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

SYSTEM_COLORS = {
    "none": "#888888",
    "nlms_f64": "#1f77b4",
    "speex": "#ff7f0e",
    "nlms_q15": "#d62728",
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
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for system in systems:
        m = np.load(cell_dir / f"metrics_{system}.npz")
        color = SYSTEM_COLORS.get(system)
        ax.plot(m["frame_times_s"], m["erle_smoothed_db"], lw=1.2,
                label=system, color=color)
        ss = rows.loc[rows["system"] == system, "erle_steady_state_db"]
        if len(ss) and np.isfinite(ss.iloc[0]):
            ax.axhline(ss.iloc[0], ls=":", lw=0.8, color=color, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("smoothed ERLE (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"baseline cell (RT60 0.4 s, 1.0 m, far single-talk) — "
                 f"seed {seed}", fontsize=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_erle.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    for system in ("nlms_f64", "nlms_q15"):
        path = cell_dir / f"metrics_{system}.npz"
        if not path.exists():
            continue
        m = np.load(path)
        if "misalignment_curve_db" not in m:
            continue
        ax.plot(m["misalignment_times_s"], m["misalignment_curve_db"],
                lw=1.2, color=SYSTEM_COLORS[system], label=system)
        plotted = True
    if plotted:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("misalignment (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("baseline cell — NLMS coefficient misalignment vs "
                     "h_echo", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_misalignment.png", dpi=120,
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
                             figsize=(5.2 * len(systems), 4.2), sharey=True)
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
                            va="center", color="white", fontsize=10,
                            path_effects=[
                                matplotlib.patheffects.withStroke(
                                    linewidth=2, foreground="black")])
        ax.set_xticks(range(len(dists)), [f"{d:g}" for d in dists])
        ax.set_yticks(range(len(rts)), _rt60_row_labels(sub))
        ax.set_xlabel("speaker–mic distance (m)")
        ax.set_title(system)
    np.atleast_1d(axes)[0].set_ylabel("RT60 target (s), achieved range")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label=label, shrink=0.85)
    fig.suptitle(f"Stage A — {label} (mean over 3 seeds)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
        for k, level in enumerate(levels):
            pts = grp[grp[level_col] == level]
            if pts.empty:
                continue
            xs = k + offsets.get(system, 0.0) + \
                (pts["seed"].to_numpy() - 1) * 0.045
            ys = pts[value_col].to_numpy(dtype=float)
            ok = pts["status"].to_numpy() == "ok"
            ax.scatter(xs[ok], ys[ok], s=28, color=color, zorder=3,
                       label=system if k == 0 else None)
            ax.scatter(xs[~ok], ys[~ok], s=44, color=color, marker="x",
                       zorder=3,
                       label=f"{system} (diverged)" if (~ok).any() and k >= 0
                       and f"{system} (diverged)" not in
                       [h.get_label() for h in ax.collections] else None)
            finite = ys[np.isfinite(ys)]
            if len(finite):
                ax.hlines(np.mean(finite), k - 0.3, k + 0.3, color=color,
                          lw=1.0, alpha=0.5)
    ax.set_xticks(range(len(levels)), [str(lv) for lv in levels])
    ax.grid(alpha=0.3, axis="y")


def plot_stage_b_talk(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "talk")]
    levels = ["far_single", "double", "near_single"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _strip_axis(ax1, sub, levels, "erle_steady_state_db")
    ax1.set_ylabel("steady-state ERLE (dB)")
    ax1.set_title("echo suppression (far-only regions)", fontsize=10)
    ax1.legend(fontsize=8)
    _strip_axis(ax2, sub, levels, "segsnr_db")
    ax2.set_ylabel("segmental SNR (dB)")
    ax2.set_title("near-end distortion (near-active regions)", fontsize=10)
    fig.suptitle("Stage B — talk state (per-seed points, bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_talk.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_noise(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "noise")]
    levels = ["no_noise", "snr20", "snr10"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _strip_axis(ax, sub, levels, "erle_steady_state_db")
    ax.set_ylabel("steady-state ERLE (dB)")
    ax.set_xlabel("background noise")
    ax.legend(fontsize=8)
    ax.set_title("Stage B — noise (per-seed points, bar = mean)", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_noise.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_tail(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "tail_length")]
    levels = ["50ms", "100ms", "200ms", "400ms"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _strip_axis(ax1, sub, levels, "erle_steady_state_db")
    ax1.set_ylabel("steady-state ERLE (dB)")
    ax1.set_xlabel("filter length")
    ax1.legend(fontsize=8)
    _strip_axis(ax2, sub, levels, "convergence_time_s")
    ax2.set_ylabel("convergence time (s)")
    ax2.set_xlabel("filter length")
    fig.suptitle("Stage B — tail length (per-seed points, bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_tail.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_stage_b_mu(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    sub = df[(df["stage"] == "b") & (df["axis"] == "mu")]
    swept = sub[sub["level"] != "reference"]
    ref = sub[sub["level"] == "reference"]
    levels = sorted(swept["level"].unique(), key=lambda s: float(s[2:]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, ylabel in [(ax1, "erle_steady_state_db",
                             "steady-state ERLE (dB)"),
                            (ax2, "convergence_time_s",
                             "convergence time (s)")]:
        _strip_axis(ax, swept, levels, col)
        if not ref.empty:
            ax.axhline(ref[col].mean(), color=SYSTEM_COLORS["speex"],
                       ls="--", lw=1.2,
                       label="speex @ baseline (reference, not swept)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("NLMS step size μ")
        ax.legend(fontsize=8)
    fig.suptitle("Stage B — NLMS step size (per-seed points, bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stage_b_mu.png", dpi=120, bbox_inches="tight")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, ylabel in [(ax1, "erle_steady_state_db",
                             "steady-state ERLE (dB)"),
                            (ax2, "convergence_time_s",
                             "convergence time (s)")]:
        for k, level in enumerate(levels):
            pts = sub[sub["level"] == level]
            xs = k + (pts["seed"].to_numpy() - 1) * 0.06
            ys = pts[col].to_numpy(dtype=float)
            ok = np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], s=34, color=color, zorder=3,
                       label="nlms_q15 (masked)" if k == 0 else None)
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
                   label="nlms_f64, same scenario (float reference)")
        if col == "erle_steady_state_db":
            ax.axhline(0.0, color="#888888", ls=":", lw=1.0,
                       label="0 dB (no processing)")
        ax.set_xticks(range(len(levels)),
                      [lv.replace("bit", "") for lv in levels])
        ax.set_xlabel("effective coefficient word length (bits)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Word-length sweep — Q15 NLMS with masked coefficients "
                 "(per-seed points, bar = mean)", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "word_length_sweep.png", dpi=120,
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
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, div[::dec], lw=0.9, color=SYSTEM_COLORS["nlms_q15"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("10·log10(‖w_q15/2¹⁵ − w_float‖² / ‖w_float‖²)  (dB)")
    ax.grid(alpha=0.3)
    ax.set_title("baseline cell — Q15 coefficient divergence from the "
                 "float64 shadow filter (identical quantised input)",
                 fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "q15_float_divergence.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def plot_audibility(csv_path: Path, out_dir: Path) -> None:
    """Residual-echo audibility across the noise axis (the cells with a
    real masker) plus the double-talk cell: fraction of TF units above
    the masking threshold, and mean excess over it."""
    df = pd.read_csv(csv_path)
    noise = df[(df["stage"] == "b") & (df["axis"] == "noise")].copy()
    dt = df[(df["stage"] == "b") & (df["axis"] == "talk")
            & (df["level"] == "double")].copy()
    dt["level"] = "double_talk"
    sub = pd.concat([noise, dt])
    levels = ["no_noise", "snr20", "snr10", "double_talk"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    _strip_axis(ax1, sub, levels, "audibility_fraction")
    ax1.set_ylabel("fraction of TF units above threshold")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=8)
    _strip_axis(ax2, sub, levels, "audibility_excess_db")
    ax2.set_ylabel("mean excess above threshold (dB)")
    for ax in (ax1, ax2):
        ax.set_xlabel("condition (baseline room)")
    fig.suptitle("Residual-echo audibility — far-single segments "
                 "(per-seed points, bar = mean); threshold = spread "
                 "noise masking in snr cells, constant floor where the "
                 "masker is silent (no_noise, double_talk)", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "audibility.png", dpi=120, bbox_inches="tight")
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
