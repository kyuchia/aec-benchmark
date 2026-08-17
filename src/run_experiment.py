"""Run benchmark scenarios defined in config/scenarios.yaml.

Current stage: single-scenario end-to-end runs. For each (scenario, seed)
this builds the room, synthesises the signals, runs the requested systems,
persists every intermediate signal plus per-run metadata, and prints
sanity diagnostics. Batch orchestration over the full matrix comes later.

Usage:
    python src/run_experiment.py --scenario baseline --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import metrics
from aec_nlms import nlms
from aec_speex import run_speex_aec
from room import build_rirs
from segment import segment
from signals import (
    SignalSet,
    compute_int16_scale,
    float_to_int16,
    int16_to_float,
    synthesise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "scenarios.yaml"
RUNS_DIR = REPO_ROOT / "data" / "generated" / "runs"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # dataset_dir is relative to the repo root
    cfg["speech"]["dataset_dir"] = str(REPO_ROOT / cfg["speech"]["dataset_dir"])
    return cfg


# ---------------------------------------------------------------------------
# Systems under test (T0). Each takes float x, d and the run's shared int16
# scale, and returns (float e, system metadata, extra arrays to persist).
# ---------------------------------------------------------------------------

def run_none(x: np.ndarray, d: np.ndarray, scale: float, cfg: dict,
             sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    """0 dB ERLE reference.

    Not a pure passthrough: d goes through the identical int16 round-trip
    at the identical scale as the fixed-point system, so the reference is
    measured on the same signal path (same quantisation floor).
    """
    d16 = float_to_int16(d, scale, name="d (none)")
    return int16_to_float(d16, scale), {}, {}


def run_nlms_f64(x: np.ndarray, d: np.ndarray, scale: float, cfg: dict,
                 sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    """Float64 NLMS. Stays in float throughout — no int16 round-trip.

    That asymmetry against the fixed-point systems is deliberate and is
    noted in the report as a caveat of comparing float and fixed point.
    """
    sys_cfg = cfg["systems"]["nlms_f64"]
    L = int(round(sys_cfg["filter_length_ms"] * 1e-3 * sample_rate))
    record_every = sys_cfg.get("record_every")
    out = nlms(x, d, L=L, mu=float(sys_cfg["mu"]),
               delta=float(sys_cfg["delta"]),
               record_every=int(record_every) if record_every else None)
    extras = {"w_final": out["w_final"]}
    if out["w_traj"] is not None:
        extras["w_traj"] = out["w_traj"]
    return out["e"], {
        "filter_length_samples": L,
        "mu": sys_cfg["mu"],
        "delta": sys_cfg["delta"],
        "record_every": record_every,
    }, extras


def run_speex(x: np.ndarray, d: np.ndarray, scale: float, cfg: dict,
              sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    sys_cfg = cfg["systems"]["speex"]
    frame_size = int(sys_cfg["frame_size"])
    filter_length = int(round(
        sys_cfg["filter_length_ms"] * 1e-3 * sample_rate))
    x16 = float_to_int16(x, scale, name="x (speex)")
    d16 = float_to_int16(d, scale, name="d (speex)")
    e16 = run_speex_aec(x16, d16, frame_size, filter_length, sample_rate)
    # Speex saturates internally rather than wrapping; full-scale output
    # samples indicate the headroom was insufficient for this run.
    n_saturated = int(np.sum(np.abs(e16.astype(np.int32)) >= 32767))
    if n_saturated > 0:
        raise AssertionError(
            f"speex output saturated on {n_saturated} samples; "
            "increase int16_headroom_db"
        )
    return int16_to_float(e16, scale), {
        "frame_size": frame_size,
        "filter_length_samples": filter_length,
        "output_saturated_samples": n_saturated,
    }, {}


SYSTEM_RUNNERS = {
    "none": run_none,
    "nlms_f64": run_nlms_f64,
    "speex": run_speex,
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(cfg: dict, scenario_id: str, seed_index: int,
               systems: list[str]) -> dict:
    scenario = cfg["scenarios"][scenario_id]
    sample_rate = int(cfg["sample_rate"])

    rirs = build_rirs(cfg["room"], scenario["rt60_s"],
                      scenario["speaker_mic_distance_m"], sample_rate)
    sigs: SignalSet = synthesise(cfg, scenario, seed_index,
                                 rirs.h_echo, rirs.h_near)

    headroom_db = float(cfg["levels"]["int16_headroom_db"])
    scale = compute_int16_scale([sigs.x, sigs.d], headroom_db)

    run_dir = RUNS_DIR / scenario_id / f"seed{seed_index}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        run_dir / "signals.npz",
        x=sigs.x, d_echo=sigs.d_echo, s_clean=sigs.s_clean, s=sigs.s,
        v=sigs.v, d=sigs.d, h_echo=rirs.h_echo, h_near=rirs.h_near,
        far_mask=sigs.far_mask, near_mask=sigs.near_mask,
        sample_rate=sample_rate,
    )

    def save_wav(name: str, sig: np.ndarray) -> None:
        sf.write(run_dir / name, float_to_int16(sig, scale, name=name),
                 sample_rate, subtype="PCM_16")

    save_wav("x.wav", sigs.x)
    save_wav("d.wav", sigs.d)

    seg = segment(sigs.d_echo, sigs.s, sample_rate, cfg["segmentation"])
    np.savez_compressed(
        run_dir / "segmentation.npz",
        frame_len=seg.frame_len,
        far_active=seg.far_active,
        near_active=seg.near_active,
    )
    met_cfg = cfg["metrics"]
    frame_times = seg.frame_times_s()
    has_double_talk = bool(np.any(seg.double_talk))

    results: dict = {}
    metrics_out: dict = {}
    for system in systems:
        e, sys_meta, extras = SYSTEM_RUNNERS[system](sigs.x, sigs.d, scale,
                                                     cfg, sample_rate)
        np.savez_compressed(run_dir / f"e_{system}.npz", e=e, **extras)
        save_wav(f"e_{system}.wav", e)
        results[system] = sys_meta

        erle = metrics.erle_curve(
            sigs.d, e, seg.erle_valid, seg.frame_len,
            ema_alpha=float(met_cfg["erle_ema_alpha"]),
            steady_state_last_fraction=float(
                met_cfg["steady_state_last_fraction"]),
            sanity_max_db=float(met_cfg["erle_sanity_max_db"]),
        )
        conv_t = metrics.convergence_time_s(
            erle["erle_smoothed_db"], seg.erle_valid,
            erle["steady_state_db"], frame_times,
            float(met_cfg["convergence_fraction"]),
        )
        entry: dict = {
            "erle_steady_state_db": erle["steady_state_db"],
            "erle_n_valid_frames": erle["n_valid_frames"],
            "convergence_time_s": None if np.isnan(conv_t) else conv_t,
            "converged": bool(np.isfinite(conv_t)),
        }

        metric_arrays: dict = {
            "frame_times_s": frame_times,
            "erle_db": erle["erle_db"],
            "erle_smoothed_db": erle["erle_smoothed_db"],
        }

        if has_double_talk:
            segsnr = metrics.segmental_snr_db(sigs.s, e, seg.double_talk,
                                              seg.frame_len)
            lsd = metrics.log_spectral_distance_db(
                sigs.s, e, seg.double_talk, seg.frame_len,
                floor_db=float(met_cfg["lsd_floor_db"]))
            near_idx = np.flatnonzero(seg.near_active)
            span = slice(near_idx[0] * seg.frame_len,
                         (near_idx[-1] + 1) * seg.frame_len)
            quality = metrics.speech_quality_scores(sigs.s[span], e[span],
                                                    sample_rate)
            entry.update({
                "double_talk_segsnr_db": segsnr["segsnr_db"],
                "double_talk_segsnr_n_frames": segsnr["n_frames"],
                "double_talk_lsd_db": lsd["lsd_db"],
                **quality,
            })

        if "w_final" in extras:
            entry["misalignment_final_db"] = metrics.misalignment_db(
                extras["w_final"], rirs.h_echo)
            if "w_traj" in extras:
                record_every = int(
                    cfg["systems"][system]["record_every"])
                metric_arrays["misalignment_curve_db"] = (
                    metrics.misalignment_curve_db(extras["w_traj"],
                                                  rirs.h_echo))
                metric_arrays["misalignment_times_s"] = (
                    (np.arange(len(extras["w_traj"])) + 1)
                    * record_every / sample_rate)

        np.savez_compressed(run_dir / f"metrics_{system}.npz",
                            **metric_arrays)
        metrics_out[system] = entry

    meta = {
        "scenario_id": scenario_id,
        "scenario": scenario,
        "seed_index": seed_index,
        "sample_rate": sample_rate,
        "rt60_target_s": rirs.rt60_target_s,
        "rt60_achieved_echo_s": rirs.rt60_achieved_echo_s,
        "rt60_achieved_near_s": rirs.rt60_achieved_near_s,
        "rt60_calibration": rirs.calibration,
        "geometry": rirs.geometry,
        "int16_scale": scale,
        "int16_headroom_db": headroom_db,
        "signals": sigs.meta,
        "systems": results,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    metrics_json = {
        "scenario_id": scenario_id,
        "seed_index": seed_index,
        "has_double_talk": has_double_talk,
        "systems": metrics_out,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    meta["metrics"] = metrics_json
    return meta


def _print_diagnostics(meta: dict) -> None:
    geo = meta["geometry"]
    print(f"\n=== {meta['scenario_id']} / seed {meta['seed_index']} ===")
    print(f"RT60 target {meta['rt60_target_s']:.2f} s | achieved "
          f"echo {meta['rt60_achieved_echo_s']:.3f} s, "
          f"near {meta['rt60_achieved_near_s']:.3f} s")
    print(f"h_echo first arrival: sample {geo['h_echo_first_arrival_sample']} "
          f"(expected direct delay {geo['expected_direct_delay_samples']:.1f} "
          f"+ ISM fractional-delay offset)")
    print(f"h_echo peak: {geo['h_echo_peak_value']:.4f} at sample "
          f"{geo['h_echo_peak_sample']}")
    print(f"int16 scale: {meta['int16_scale']:.1f} "
          f"(headroom {meta['int16_headroom_db']} dB)")
    for system, entry in meta["metrics"]["systems"].items():
        conv = entry["convergence_time_s"]
        conv_str = f"{conv:.2f} s" if conv is not None else "not converged"
        line = (f"{system:>8}: steady-state ERLE "
                f"{entry['erle_steady_state_db']:5.1f} dB | "
                f"convergence {conv_str}")
        if "double_talk_segsnr_db" in entry:
            line += f" | DT segSNR {entry['double_talk_segsnr_db']:.1f} dB"
        if "misalignment_final_db" in entry:
            line += f" | misalign {entry['misalignment_final_db']:.1f} dB"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+",
                        default=["none", "nlms_f64", "speex"],
                        choices=sorted(SYSTEM_RUNNERS))
    args = parser.parse_args()

    cfg = load_config()
    if args.scenario not in cfg["scenarios"]:
        parser.error(f"unknown scenario {args.scenario!r}; defined: "
                     f"{sorted(cfg['scenarios'])}")
    meta = run_single(cfg, args.scenario, args.seed, args.systems)
    _print_diagnostics(meta)


if __name__ == "__main__":
    main()
