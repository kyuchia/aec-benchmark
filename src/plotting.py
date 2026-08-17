"""Figure generation. One function per figure type.

Every function reads previously computed arrays/scalars from a run
directory (signals.npz, segmentation.npz, metrics_<system>.npz,
metrics.json) — nothing is recomputed here. Figures for the report land
in results/figures/ only when produced from real metric runs.
"""

from __future__ import annotations

import json
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
}


def _load_metrics_json(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json") as f:
        return json.load(f)


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


def plot_erle_overlay(run_dir: Path, out_path: Path,
                      title: str | None = None) -> None:
    """Smoothed ERLE curves for every system with metrics in the run dir."""
    meta = _load_metrics_json(run_dir)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for system, entry in meta["systems"].items():
        m = np.load(run_dir / f"metrics_{system}.npz")
        color = SYSTEM_COLORS.get(system)
        ax.plot(m["frame_times_s"], m["erle_smoothed_db"], lw=1.2,
                label=system, color=color)
        ss = entry["erle_steady_state_db"]
        if ss is not None and np.isfinite(ss):
            ax.axhline(ss, ls=":", lw=0.8, color=color, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("smoothed ERLE (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_bar(run_dir: Path, out_path: Path,
                         title: str | None = None) -> None:
    """Convergence time per system; non-converged shown as hatched marker."""
    meta = _load_metrics_json(run_dir)
    systems, values = [], []
    for system, entry in meta["systems"].items():
        systems.append(system)
        v = entry["convergence_time_s"]
        values.append(np.nan if v is None else v)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [SYSTEM_COLORS.get(s, "#333333") for s in systems]
    finite = [0.0 if np.isnan(v) else v for v in values]
    bars = ax.bar(systems, finite, color=colors)
    for bar, v in zip(bars, values):
        if np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "not\nconverged", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("convergence time (s)")
    ax.grid(alpha=0.3, axis="y")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_misalignment(run_dir: Path, out_path: Path,
                      title: str | None = None) -> None:
    """Misalignment trajectory for systems that recorded coefficients."""
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    meta = _load_metrics_json(run_dir)
    for system in meta["systems"]:
        path = run_dir / f"metrics_{system}.npz"
        m = np.load(path)
        if "misalignment_curve_db" not in m:
            continue
        ax.plot(m["misalignment_times_s"], m["misalignment_curve_db"],
                lw=1.2, label=system, color=SYSTEM_COLORS.get(system))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("time (s)")
    ax.set_ylabel("misalignment (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    offsets = {"none": -0.22, "nlms_f64": 0.0, "speex": 0.22}
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


def batch_figures(csv_path: Path, out_dir: Path) -> None:
    plot_stage_a(csv_path, out_dir)
    plot_stage_b_talk(csv_path, out_dir)
    plot_stage_b_noise(csv_path, out_dir)
    plot_stage_b_tail(csv_path, out_dir)
    plot_stage_b_mu(csv_path, out_dir)
