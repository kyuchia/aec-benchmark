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

from aec_speex import run_speex_aec
from room import build_rirs
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
# scale, and returns float e plus a dict of system-specific metadata.
# ---------------------------------------------------------------------------

def run_none(x: np.ndarray, d: np.ndarray, scale: float, cfg: dict,
             sample_rate: int) -> tuple[np.ndarray, dict]:
    """0 dB ERLE reference.

    Not a pure passthrough: d goes through the identical int16 round-trip
    at the identical scale as the fixed-point system, so the reference is
    measured on the same signal path (same quantisation floor).
    """
    d16 = float_to_int16(d, scale, name="d (none)")
    return int16_to_float(d16, scale), {}


def run_speex(x: np.ndarray, d: np.ndarray, scale: float, cfg: dict,
              sample_rate: int) -> tuple[np.ndarray, dict]:
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
    }


SYSTEM_RUNNERS = {
    "none": run_none,
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

    results: dict = {}
    for system in systems:
        e, sys_meta = SYSTEM_RUNNERS[system](sigs.x, sigs.d, scale, cfg,
                                             sample_rate)
        np.savez_compressed(run_dir / f"e_{system}.npz", e=e)
        save_wav(f"e_{system}.wav", e)

        # Rough echo-reduction diagnostic (NOT the ERLE metric): energy in
        # the final third, where adaptation has had time to settle. Only
        # indicative for far-single-talk runs.
        tail = slice(int(len(sigs.d) * 2 / 3), None)
        p_d = float(np.mean(sigs.d[tail] ** 2))
        p_e = float(np.mean(e[tail] ** 2))
        results[system] = {
            "tail_echo_reduction_db": 10.0 * np.log10(p_d / p_e),
            **sys_meta,
        }

    meta = {
        "scenario_id": scenario_id,
        "scenario": scenario,
        "seed_index": seed_index,
        "sample_rate": sample_rate,
        "rt60_target_s": rirs.rt60_target_s,
        "rt60_achieved_echo_s": rirs.rt60_achieved_echo_s,
        "rt60_achieved_near_s": rirs.rt60_achieved_near_s,
        "geometry": rirs.geometry,
        "int16_scale": scale,
        "int16_headroom_db": headroom_db,
        "signals": sigs.meta,
        "systems": results,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
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
    for system, res in meta["systems"].items():
        print(f"{system:>8}: tail echo reduction "
              f"{res['tail_echo_reduction_db']:+.1f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=["none", "speex"],
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
