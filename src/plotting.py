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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
