"""Batch entry point: run the experiment matrix from config/scenarios.yaml.

One row in results/raw/runs.csv = one (scenario, seed, system) triple,
carrying every metric plus provenance (achieved RT60, scaling constant,
calibrated absorption, git SHA, wall time). Figures and the report
aggregate from that file only.

Determinism: run IDs are pure functions of (stage, axis, level, seed,
system). Utterance selection is keyed by the seed index (so speech
material is paired across scenario levels — factor effects are not
confounded with speaker changes), and the noise realisation is seeded
from the (scenario, seed) cell identity — identical for every system on
that cell, which is the fairness condition, and stable across re-runs.

Failure policy: a failed or diverged run writes its row with status
failed/diverged plus the reason, and the batch continues. After the
batch, the row count is asserted against the expected count. Divergence
(unprotected NLMS in double-talk is the expected case) is detected, not
prevented, and its onset time is recorded as a data point.

Usage:
    python src/run_experiment.py --batch            # full matrix
    python src/run_experiment.py --scenario baseline --seed 0   # one cell
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import metrics
import psychoacoustic
from aec_nlms import nlms
from aec_nlms_fixed import nlms_q15
from aec_speex import run_speex_aec
from room import RoomRirs, build_rirs
from segment import Segmentation, segment
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
RAW_DIR = REPO_ROOT / "results" / "raw"

# |e| exceeding this factor times peak |d| (or any non-finite sample)
# counts as divergence; the first offending sample sets the onset time.
DIVERGENCE_FACTOR = 10.0

CSV_FIELDS = [
    "run_id", "stage", "axis", "level", "scenario_key", "seed", "system",
    "status", "fail_reason", "divergence_time_s",
    "talk", "noise_type", "snr_db", "ser_db",
    "rt60_target_s", "rt60_achieved_echo_s", "rt60_achieved_near_s",
    "speaker_mic_distance_m", "absorption_calibrated",
    "absorption_sabine_init",
    "filter_length_ms", "mu", "frame_size",
    "int16_scale", "int16_headroom_db",
    "erle_steady_state_db", "erle_n_valid_frames",
    "convergence_time_s", "converged",
    "segsnr_db", "stoi", "pesq_wb", "lsd_db",
    "misalignment_final_db",
    "audibility_fraction", "audibility_excess_db", "audibility_n_units",
    "audibility_fraction_traj", "audibility_fraction_interp",
    "recon_err_db", "recon_err_interp_db",
    "speex_tworun_erle_diff_db",
    "mu_q15", "delta_q30", "coeff_bits",
    "n_stall_events", "stall_first_time_s", "n_tap_stalls",
    "n_sat_total", "n_sat_y", "n_sat_err", "n_sat_gain", "n_sat_coeff",
    "n_sat_coeff_taps", "sat_first_time_s",
    "coeff_div_final_db", "coeff_div_steady_db",
    "far_speaker", "near_speaker",
    "wall_time_s", "git_sha",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["speech"]["dataset_dir"] = str(REPO_ROOT / cfg["speech"]["dataset_dir"])
    return cfg


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=REPO_ROOT,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, cwd=REPO_ROOT,
                               check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Systems under test. Each takes float x, d, the run's shared int16 scale
# and its merged parameter dict, and returns (float e, metadata, extras).
# ---------------------------------------------------------------------------

def run_none(x: np.ndarray, d: np.ndarray, scale: float, sys_params: dict,
             sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    """0 dB ERLE reference.

    Not a pure passthrough: d goes through the identical int16 round-trip
    at the identical scale as the fixed-point system, so the reference is
    measured on the same signal path (same quantisation floor).
    """
    d16 = float_to_int16(d, scale, name="d (none)")
    return int16_to_float(d16, scale), {}, {}


def run_nlms_f64(x: np.ndarray, d: np.ndarray, scale: float,
                 sys_params: dict,
                 sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    """Float64 NLMS. Stays in float throughout — no int16 round-trip.

    That asymmetry against the fixed-point systems is deliberate and is
    noted in the report as a caveat of comparing float and fixed point.
    """
    L = int(round(sys_params["filter_length_ms"] * 1e-3 * sample_rate))
    record_every = sys_params.get("record_every")
    out = nlms(x, d, L=L, mu=float(sys_params["mu"]),
               delta=float(sys_params["delta"]),
               record_every=int(record_every) if record_every else None)
    extras = {"w_final": out["w_final"]}
    if out["w_traj"] is not None:
        extras["w_traj"] = out["w_traj"]
    return out["e"], {
        "filter_length_ms": sys_params["filter_length_ms"],
        "mu": sys_params["mu"],
    }, extras


def run_speex(x: np.ndarray, d: np.ndarray, scale: float, sys_params: dict,
              sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    frame_size = int(sys_params["frame_size"])
    filter_length = int(round(
        sys_params["filter_length_ms"] * 1e-3 * sample_rate))
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
        "filter_length_ms": sys_params["filter_length_ms"],
        "frame_size": frame_size,
    }, {}


def run_nlms_q15(x: np.ndarray, d: np.ndarray, scale: float,
                 sys_params: dict,
                 sample_rate: int) -> tuple[np.ndarray, dict, dict]:
    """Q15 fixed-point NLMS on the run's shared int16 path.

    x and d are quantised with the identical scaling constant as speex
    and none. Since both are scaled by the same factor, the echo path in
    the int16 domain is unchanged, so w/2^15 estimates h_echo directly
    (no rescaling before misalignment). The in-loop float64 shadow
    filter sees the same quantised input, isolating arithmetic
    degradation from input quantisation in the divergence curve.
    """
    L = int(round(sys_params["filter_length_ms"] * 1e-3 * sample_rate))
    mu_q15 = int(round(float(sys_params["mu"]) * 32768))
    delta_q30 = max(1, int(round(float(sys_params["delta"]) * (1 << 30))))
    coeff_bits = int(sys_params.get("coeff_bits", 15))
    record_every = sys_params.get("record_every")
    x16 = float_to_int16(x, scale, name="x (nlms_q15)")
    d16 = float_to_int16(d, scale, name="d (nlms_q15)")
    out = nlms_q15(x16, d16, L=L, mu_q15=mu_q15, delta_q30=delta_q30,
                   coeff_bits=coeff_bits,
                   record_every=int(record_every) if record_every else None,
                   shadow_float=True)

    extras = {
        "w_final": out["w_final"] / 32768.0,  # int16-domain estimate of h
        "w_final_q15": out["w_final"],
        "stall_positions": out["stall_positions"],
        "tap_stalls_per_sample": out["tap_stalls_per_sample"],
        "coeff_div_db": out["coeff_div_db"],
    }
    for site, pos in out["sat_positions"].items():
        extras[f"sat_positions_{site}"] = pos
    if out["w_traj"] is not None:
        extras["w_traj"] = out["w_traj"] / 32768.0
        extras["w_traj_q15"] = out["w_traj"]

    sc = out["sat_counts"]
    all_sat = np.concatenate(list(out["sat_positions"].values()))
    div = out["coeff_div_db"][np.isfinite(out["coeff_div_db"])]
    n_steady = max(1, int(round(0.3 * len(div))))

    def first_time_s(pos: np.ndarray) -> float | None:
        return float(pos.min() / sample_rate) if len(pos) else None

    meta = {
        "filter_length_ms": sys_params["filter_length_ms"],
        "mu": sys_params["mu"],
        "mu_q15": mu_q15,
        "delta_q30": delta_q30,
        "coeff_bits": coeff_bits,
        "n_stall_events": out["n_stall_events"],
        "stall_first_time_s": first_time_s(out["stall_positions"]),
        "n_tap_stalls": out["n_tap_stalls"],
        "n_sat_total": sum(sc.values()),
        "n_sat_y": sc["y"],
        "n_sat_err": sc["err"],
        "n_sat_gain": sc["gain"],
        "n_sat_coeff": sc["coeff"],
        "n_sat_coeff_taps": out["n_sat_coeff_taps"],
        "sat_first_time_s": first_time_s(all_sat),
        "coeff_div_final_db": float(div[-1]) if len(div) else None,
        "coeff_div_steady_db": (float(np.median(div[-n_steady:]))
                                if len(div) else None),
    }
    return int16_to_float(out["e"], scale), meta, extras


SYSTEM_RUNNERS = {
    "none": run_none,
    "nlms_f64": run_nlms_f64,
    "speex": run_speex,
    "nlms_q15": run_nlms_q15,
}


def merged_sys_params(cfg: dict, system: str, overrides: dict) -> dict:
    params = dict(cfg["systems"][system])
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Residual-echo isolation + perceptual audibility (spec §8.7)
# ---------------------------------------------------------------------------

def _erle_valid_samples(seg: Segmentation, n: int) -> np.ndarray:
    """Per-sample mask of ERLE-valid (far-single) frames."""
    mask = np.repeat(seg.erle_valid, seg.frame_len)
    if len(mask) < n:
        mask = np.concatenate([mask, np.zeros(n - len(mask), bool)])
    return mask[:n]


def residual_and_audibility(system: str, sigs: SignalSet, seg: Segmentation,
                            scale: float, sys_params: dict, extras: dict,
                            e: np.ndarray, sample_rate: int,
                            aud_cfg: dict, met_cfg: dict) -> tuple[dict,
                                                                   dict]:
    """Isolate each system's residual echo and score its audibility.

    Primary residual, linear-subtraction systems (none / nlms_f64 /
    nlms_q15): the exact component identity r = e - s - v. For a
    canceller with output e = d - y and d = d_echo + s + v this is
    algebraically identical to the spec's decomposition d_echo - y(w(n))
    evaluated with the *exact per-sample* coefficients — i.e. the
    trajectory method at record_every = 1, which is the convention the
    reconstruction is unit-tested exact at. (Exact for nlms_f64; for
    nlms_q15 exact except at error-narrowing saturation samples, whose
    count the instrumentation records.)

    The recorded-trajectory reconstruction (hold = block-start state;
    f64 additionally under linear interpolation; Q15 in Q15 arithmetic
    from the int16 trajectory) is still computed: its error against the
    exact y = d - e output is the QC column, and audibility fractions
    from the trajectory residuals are recorded alongside the primary
    ones so the decimation's materiality is measured, not assumed.

    speex: the spec's two-run approximation (same configuration on an
    echo-only microphone signal), with the ERLE difference between the
    two runs recorded as the approximation's error bar. The component
    identity is deliberately NOT used for speex: its internal DC notch
    on the microphone path means e != d - y exactly, so an identity
    residual would carry notch artifacts of s + v.

    The masker is near-end speech plus noise (s + v); audibility is
    computed over ERLE-valid frames only, matching the ERLE metric.
    """
    entry: dict = {}
    arrays: dict = {}
    valid_samp = _erle_valid_samples(seg, len(sigs.d))
    record_every = sys_params.get("record_every")

    def traj_fraction(residual_traj: np.ndarray) -> float:
        return psychoacoustic.audibility(
            residual_traj, sigs.s + sigs.v, seg.erle_valid,
            seg.frame_len, sample_rate, aud_cfg)["fraction"]

    if system == "none":
        residual = e - sigs.s - sigs.v
    elif system == "nlms_f64":
        residual = e - sigs.s - sigs.v
        y_act = sigs.d - e
        y_hold = psychoacoustic.reconstruct_output_f64(
            sigs.x, extras["w_traj"], int(record_every))
        y_interp = psychoacoustic.reconstruct_output_f64(
            sigs.x, extras["w_traj"], int(record_every), interpolate=True)
        entry["recon_err_db"] = psychoacoustic.reconstruction_error_db(
            y_hold, y_act, valid_samp)
        entry["recon_err_interp_db"] = \
            psychoacoustic.reconstruction_error_db(y_interp, y_act,
                                                   valid_samp)
        entry["audibility_fraction_traj"] = traj_fraction(
            sigs.d_echo - y_hold)
        entry["audibility_fraction_interp"] = traj_fraction(
            sigs.d_echo - y_interp)
    elif system == "nlms_q15":
        residual = e - sigs.s - sigs.v
        x16 = float_to_int16(sigs.x, scale, name="x (nlms_q15 recon)")
        d16 = float_to_int16(sigs.d, scale, name="d (nlms_q15 recon)")
        e16 = np.round(e * scale).astype(np.int64)
        y_act = d16.astype(np.int64) - e16
        ok = valid_samp.copy()
        ok[extras["sat_positions_err"]] = False  # d-e != y where err clipped
        y_rec = psychoacoustic.reconstruct_output_q15(
            x16, extras["w_traj_q15"], int(record_every))
        entry["recon_err_db"] = psychoacoustic.reconstruction_error_db(
            y_rec, y_act, ok)
        entry["audibility_fraction_traj"] = traj_fraction(
            sigs.d_echo - y_rec / scale)
    elif system == "speex":
        d_echo16 = float_to_int16(sigs.d_echo, scale,
                                  name="d_echo (speex two-run)")
        x16 = float_to_int16(sigs.x, scale, name="x (speex two-run)")
        e2 = run_speex_aec(x16, d_echo16, int(sys_params["frame_size"]),
                           int(round(sys_params["filter_length_ms"]
                                     * 1e-3 * sample_rate)), sample_rate)
        residual = int16_to_float(e2, scale)
        # Approximation error bar: steady-state ERLE of the echo-only
        # run minus the primary run's (computed by the caller and merged
        # there, since the primary value lives in the metrics entry).
        erle2 = metrics.erle_curve(
            sigs.d_echo, residual, seg.erle_valid, seg.frame_len,
            ema_alpha=float(met_cfg["erle_ema_alpha"]),
            steady_state_last_fraction=float(
                met_cfg["steady_state_last_fraction"]),
            sanity_max_db=float(met_cfg["erle_sanity_max_db"]),
        )
        entry["_tworun_erle_steady_db"] = erle2["steady_state_db"]
    else:
        raise ValueError(f"unknown system {system!r}")

    aud = psychoacoustic.audibility(residual, sigs.s + sigs.v,
                                    seg.erle_valid, seg.frame_len,
                                    sample_rate, aud_cfg)
    entry["audibility_fraction"] = aud["fraction"]
    entry["audibility_excess_db"] = aud["excess_db"]
    entry["audibility_n_units"] = aud["n_units"]
    arrays["audibility_frame_fraction"] = aud["frame_fraction"]
    return entry, arrays


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def detect_divergence(e: np.ndarray, d: np.ndarray,
                      sample_rate: int) -> float | None:
    """First time (s) where e goes non-finite or exceeds
    DIVERGENCE_FACTOR * peak |d|; None if it never does."""
    peak_d = float(np.max(np.abs(d)))
    bad = ~np.isfinite(e) | (np.abs(e) > DIVERGENCE_FACTOR * peak_d)
    idx = np.flatnonzero(bad)
    if len(idx) == 0:
        return None
    return float(idx[0] / sample_rate)


# ---------------------------------------------------------------------------
# Per-cell preparation (shared across systems) and per-row processing
# ---------------------------------------------------------------------------

def scenario_key(scenario: dict) -> str:
    snr = scenario["snr_db"]
    return (f"rt{scenario['rt60_s']:g}_d{scenario['speaker_mic_distance_m']:g}"
            f"_{scenario['talk']}_{scenario['noise_type']}"
            f"{'' if snr is None else f'_snr{snr:g}'}"
            f"_ser{scenario['ser_db']:g}")


def cell_noise_seed(key: str, seed: int) -> int:
    return int.from_bytes(sha256(f"{key}|{seed}".encode()).digest()[:4], "big")


class BatchContext:
    """In-memory caches: RIRs per (rt60, distance), signals per cell.

    The signal cache is bounded (specs visit cells contiguously, so a
    small window suffices); RIRs are tiny and kept for the whole batch.
    """

    _MAX_SIGNAL_CELLS = 4

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._rirs: dict = {}
        self._signals: dict = {}

    def rirs(self, scenario: dict) -> RoomRirs:
        k = (scenario["rt60_s"], scenario["speaker_mic_distance_m"])
        if k not in self._rirs:
            self._rirs[k] = build_rirs(self.cfg["room"], k[0], k[1],
                                       int(self.cfg["sample_rate"]))
        return self._rirs[k]

    def signals(self, scenario: dict, seed: int) -> tuple[SignalSet, RoomRirs,
                                                          Segmentation, float,
                                                          Path]:
        key = scenario_key(scenario)
        k = (key, seed)
        if k not in self._signals:
            while len(self._signals) >= self._MAX_SIGNAL_CELLS:
                self._signals.pop(next(iter(self._signals)))
            cfg = self.cfg
            sample_rate = int(cfg["sample_rate"])
            rirs = self.rirs(scenario)
            sigs = synthesise(cfg, scenario, seed, rirs.h_echo, rirs.h_near,
                              noise_seed=cell_noise_seed(key, seed))
            scale = compute_int16_scale(
                [sigs.x, sigs.d], float(cfg["levels"]["int16_headroom_db"]))
            seg = segment(sigs.d_echo, sigs.s, sample_rate,
                          cfg["segmentation"])

            cell_dir = RUNS_DIR / key / f"seed{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cell_dir / "signals.npz",
                x=sigs.x, d_echo=sigs.d_echo, s_clean=sigs.s_clean, s=sigs.s,
                v=sigs.v, d=sigs.d, h_echo=rirs.h_echo, h_near=rirs.h_near,
                far_mask=sigs.far_mask, near_mask=sigs.near_mask,
                sample_rate=sample_rate,
            )
            np.savez_compressed(
                cell_dir / "segmentation.npz", frame_len=seg.frame_len,
                far_active=seg.far_active, near_active=seg.near_active)
            for name, sig in [("x.wav", sigs.x), ("d.wav", sigs.d)]:
                sf.write(cell_dir / name, float_to_int16(sig, scale, name),
                         sample_rate, subtype="PCM_16")
            with open(cell_dir / "cell_meta.json", "w") as f:
                json.dump({
                    "scenario": scenario, "seed": seed,
                    "int16_scale": scale,
                    "rt60_target_s": rirs.rt60_target_s,
                    "rt60_achieved_echo_s": rirs.rt60_achieved_echo_s,
                    "rt60_achieved_near_s": rirs.rt60_achieved_near_s,
                    "calibration": rirs.calibration,
                    "geometry": rirs.geometry,
                    "signals": sigs.meta,
                }, f, indent=2)
            self._signals[k] = (sigs, rirs, seg, scale, cell_dir)
        return self._signals[k]


def system_metrics(cfg: dict, sigs: SignalSet, seg: Segmentation,
                   e: np.ndarray, extras: dict, rirs: RoomRirs,
                   sample_rate: int, record_every: int | None
                   ) -> tuple[dict, dict]:
    """Compute the full metric set for one system output."""
    met_cfg = cfg["metrics"]
    frame_times = seg.frame_times_s()
    erle = metrics.erle_curve(
        sigs.d, e, seg.erle_valid, seg.frame_len,
        ema_alpha=float(met_cfg["erle_ema_alpha"]),
        steady_state_last_fraction=float(met_cfg["steady_state_last_fraction"]),
        sanity_max_db=float(met_cfg["erle_sanity_max_db"]),
    )
    conv_t = metrics.convergence_time_s(
        erle["erle_smoothed_db"], seg.erle_valid, erle["steady_state_db"],
        frame_times, float(met_cfg["convergence_fraction"]))

    entry: dict = {
        "erle_steady_state_db": erle["steady_state_db"],
        "erle_n_valid_frames": erle["n_valid_frames"],
        "convergence_time_s": None if np.isnan(conv_t) else conv_t,
        "converged": bool(np.isfinite(conv_t)),
    }
    arrays: dict = {
        "frame_times_s": frame_times,
        "erle_db": erle["erle_db"],
        "erle_smoothed_db": erle["erle_smoothed_db"],
    }

    # Near-end distortion: over double-talk frames when both sources are
    # active; for near-single-talk (no far activity at all) over
    # near-active frames, where it degenerates to plain passthrough
    # distortion. The condition is identifiable from the scenario columns.
    if np.any(seg.double_talk):
        distortion_mask = seg.double_talk
    elif np.any(seg.near_active) and not np.any(seg.far_active):
        distortion_mask = seg.near_active
    else:
        distortion_mask = None
    if distortion_mask is not None:
        segsnr = metrics.segmental_snr_db(sigs.s, e, distortion_mask,
                                          seg.frame_len)
        lsd = metrics.log_spectral_distance_db(
            sigs.s, e, distortion_mask, seg.frame_len,
            floor_db=float(met_cfg["lsd_floor_db"]))
        near_idx = np.flatnonzero(seg.near_active)
        span = slice(near_idx[0] * seg.frame_len,
                     (near_idx[-1] + 1) * seg.frame_len)
        quality = metrics.speech_quality_scores(sigs.s[span], e[span],
                                                sample_rate)
        entry.update({
            "segsnr_db": segsnr["segsnr_db"],
            "lsd_db": lsd["lsd_db"],
            "stoi": quality.get("stoi"),
            "pesq_wb": quality.get("pesq_wb"),
        })

    if "w_final" in extras:
        entry["misalignment_final_db"] = metrics.misalignment_db(
            extras["w_final"], rirs.h_echo)
        if "w_traj" in extras and record_every:
            arrays["misalignment_curve_db"] = metrics.misalignment_curve_db(
                extras["w_traj"], rirs.h_echo)
            arrays["misalignment_times_s"] = (
                (np.arange(len(extras["w_traj"])) + 1)
                * record_every / sample_rate)
    return entry, arrays


def process_row(ctx: BatchContext, spec: dict, sha: str) -> dict:
    """Run one (scenario, seed, system) triple; always returns a CSV row."""
    cfg = ctx.cfg
    sample_rate = int(cfg["sample_rate"])
    scenario = spec["scenario"]
    key = scenario_key(scenario)
    row: dict = {
        "run_id": spec["run_id"],
        "stage": spec["stage"],
        "axis": spec["axis"],
        "level": spec["level"],
        "scenario_key": key,
        "seed": spec["seed"],
        "system": spec["system"],
        "status": "ok",
        "fail_reason": "",
        "talk": scenario["talk"],
        "noise_type": scenario["noise_type"],
        "snr_db": scenario["snr_db"],
        "ser_db": scenario["ser_db"],
        "rt60_target_s": scenario["rt60_s"],
        "speaker_mic_distance_m": scenario["speaker_mic_distance_m"],
        "git_sha": sha,
    }
    t0 = time.perf_counter()
    try:
        sigs, rirs, seg, scale, cell_dir = ctx.signals(scenario, spec["seed"])
        sys_params = merged_sys_params(cfg, spec["system"],
                                       spec.get("overrides", {}))
        # The coefficient trajectory is recorded in-memory for every
        # nlms run — audibility's residual isolation needs it — but
        # persisted to the cell npz only for record_traj (baseline)
        # rows, as before.
        row.update({
            "rt60_achieved_echo_s": rirs.rt60_achieved_echo_s,
            "rt60_achieved_near_s": rirs.rt60_achieved_near_s,
            "absorption_calibrated": rirs.calibration[
                "absorption_calibrated"],
            "absorption_sabine_init": rirs.calibration[
                "absorption_sabine_init"],
            "int16_scale": scale,
            "int16_headroom_db": cfg["levels"]["int16_headroom_db"],
            "filter_length_ms": sys_params.get("filter_length_ms"),
            "mu": sys_params.get("mu")
            if spec["system"] in ("nlms_f64", "nlms_q15") else None,
            "frame_size": sys_params.get("frame_size")
            if spec["system"] == "speex" else None,
            "far_speaker": sigs.meta.get("far_speaker"),
            "near_speaker": sigs.meta.get("near_speaker"),
        })

        e, sys_meta, extras = SYSTEM_RUNNERS[spec["system"]](
            sigs.x, sigs.d, scale, sys_params, sample_rate)
        row.update({k: v for k, v in sys_meta.items() if k in CSV_FIELDS})

        label = spec["output_label"]
        persist = extras if spec.get("record_traj", False) else {
            k: v for k, v in extras.items()
            if k not in ("w_traj", "w_traj_q15")}
        np.savez_compressed(cell_dir / f"e_{label}.npz", e=e, **persist)

        div_t = detect_divergence(e, sigs.d, sample_rate)
        if div_t is not None:
            row["status"] = "diverged"
            row["divergence_time_s"] = div_t

        entry, arrays = system_metrics(
            cfg, sigs, seg, e, extras, rirs, sample_rate,
            sys_params.get("record_every"))
        aud_entry, aud_arrays = residual_and_audibility(
            spec["system"], sigs, seg, scale, sys_params, extras, e,
            sample_rate, cfg["audibility"], cfg["metrics"])
        if "_tworun_erle_steady_db" in aud_entry:
            two = aud_entry.pop("_tworun_erle_steady_db")
            base_erle = entry.get("erle_steady_state_db")
            aud_entry["speex_tworun_erle_diff_db"] = (
                two - base_erle
                if two is not None and base_erle is not None
                and np.isfinite(two) and np.isfinite(base_erle) else None)
        entry.update(aud_entry)
        arrays.update(aud_arrays)
        np.savez_compressed(cell_dir / f"metrics_{label}.npz", **arrays)
        row.update({k: v for k, v in entry.items() if k in CSV_FIELDS})
    except Exception as exc:  # noqa: BLE001 - row records the failure
        if row["status"] == "ok":
            row["status"] = "failed"
        row["fail_reason"] = f"{type(exc).__name__}: {exc}"
    row["wall_time_s"] = round(time.perf_counter() - t0, 3)
    return row


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------

def _make_spec(stage: str, axis: str, level: str, scenario: dict, seed: int,
               system: str, overrides: dict | None = None,
               record_traj: bool = False) -> dict:
    overrides = overrides or {}
    suffix = "".join(f"_{k}{v:g}" if isinstance(v, (int, float)) else ""
                     for k, v in sorted(overrides.items()))
    return {
        "stage": stage,
        "axis": axis,
        "level": level,
        "scenario": scenario,
        "seed": seed,
        "system": system,
        "overrides": overrides,
        "record_traj": record_traj,
        "output_label": f"{system}{suffix}",
        "run_id": f"{stage}.{axis}.{level}.s{seed}.{system}{suffix}",
    }


def expand_batch(cfg: dict) -> list[dict]:
    batch = cfg["batch"]
    defaults = cfg["scenario_defaults"]
    n_seeds = int(cfg["speech"]["n_seeds"])
    specs: list[dict] = []

    def record_traj(scenario: dict, system: str, overrides: dict) -> bool:
        # Trajectory recording is tied to the baseline cell with default
        # system parameters. Every spec matching it must set the flag,
        # because equal cells share one output file — a later run of the
        # same (cell, system, params) without the flag would overwrite the
        # trajectory away.
        return (system in ("nlms_f64", "nlms_q15") and not overrides
                and scenario == defaults)

    a = batch["stage_a"]
    for rt60 in a["rt60_levels_s"]:
        for dist in a["distance_levels_m"]:
            scenario = dict(a["scenario"])
            scenario["rt60_s"] = rt60
            scenario["speaker_mic_distance_m"] = dist
            level = f"rt{rt60:g}_d{dist:g}"
            for seed in range(n_seeds):
                for system in a["systems"]:
                    specs.append(_make_spec(
                        "a", "rt60_distance", level, scenario, seed, system,
                        record_traj=record_traj(scenario, system, {})))

    b = batch["stage_b"]
    for talk in b["talk"]["levels"]:
        scenario = dict(defaults)
        scenario["talk"] = talk
        for seed in range(n_seeds):
            for system in b["talk"]["systems"]:
                specs.append(_make_spec(
                    "b", "talk", talk, scenario, seed, system,
                    record_traj=record_traj(scenario, system, {})))

    for lvl in b["noise"]["levels"]:
        scenario = dict(defaults)
        scenario["noise_type"] = lvl["noise_type"]
        scenario["snr_db"] = lvl["snr_db"]
        level = ("no_noise" if lvl["noise_type"] == "none"
                 else f"snr{lvl['snr_db']:g}")
        for seed in range(n_seeds):
            for system in b["noise"]["systems"]:
                specs.append(_make_spec(
                    "b", "noise", level, scenario, seed, system,
                    record_traj=record_traj(scenario, system, {})))

    for ms in b["tail_length"]["levels_ms"]:
        scenario = dict(defaults)
        for seed in range(n_seeds):
            for system in b["tail_length"]["systems"]:
                specs.append(_make_spec(
                    "b", "tail_length", f"{ms:g}ms", scenario, seed, system,
                    overrides={"filter_length_ms": ms}))

    for mu in b["mu"]["levels"]:
        scenario = dict(defaults)
        for seed in range(n_seeds):
            for system in b["mu"]["systems"]:
                specs.append(_make_spec("b", "mu", f"mu{mu:g}", scenario,
                                        seed, system, overrides={"mu": mu}))
    for seed in range(n_seeds):
        for system in b["mu"].get("reference_systems", []):
            specs.append(_make_spec("b", "mu", "reference", dict(defaults),
                                    seed, system))

    # Word-length sweep: every level, including the unmasked 15, carries an
    # explicit coeff_bits override so the four sweep rows are constructed
    # identically (the 15-bit row duplicates the baseline nlms_q15 run
    # under its own label rather than aliasing it).
    for bits in b["word_length"]["bits"]:
        scenario = dict(defaults)
        for seed in range(n_seeds):
            for system in b["word_length"]["systems"]:
                specs.append(_make_spec(
                    "b", "word_length", f"{bits}bit", scenario, seed,
                    system, overrides={"coeff_bits": bits}))

    run_ids = [s["run_id"] for s in specs]
    if len(run_ids) != len(set(run_ids)):
        raise AssertionError("duplicate run IDs in expanded batch")
    return specs


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(cfg: dict) -> Path:
    specs = expand_batch(cfg)
    sha = git_sha()
    out_csv = RAW_DIR / "runs.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    ctx = BatchContext(cfg)
    n_bad = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="")
        writer.writeheader()
        for i, spec in enumerate(specs, 1):
            row = process_row(ctx, spec, sha)
            writer.writerow(row)
            f.flush()
            marker = "" if row["status"] == "ok" else \
                f"  [{row['status'].upper()}] {row['fail_reason']}"
            print(f"[{i:3d}/{len(specs)}] {row['run_id']:<50} "
                  f"{row['wall_time_s']:6.2f}s{marker}")
            if row["status"] != "ok":
                n_bad += 1

    total_s = time.perf_counter() - t_start
    with open(out_csv) as f:
        n_rows = sum(1 for _ in f) - 1
    if n_rows != len(specs):
        raise AssertionError(
            f"CSV has {n_rows} rows, expected {len(specs)} — silent gap")

    _write_calibration_csv(ctx)
    print(f"\nbatch complete: {n_rows} rows ({n_bad} not ok) in "
          f"{total_s / 60:.1f} min -> {out_csv}")
    return out_csv


def _write_calibration_csv(ctx: BatchContext) -> None:
    """RT60 calibration provenance, one row per level, for the report."""
    fields = ["rt60_target_s", "absorption_sabine_init",
              "rt60_achieved_sabine_init_s", "absorption_calibrated",
              "rt60_achieved_calibrated_s", "reference_distance_m",
              "tolerance_pct", "max_order"]
    by_target = {r.calibration["rt60_target_s"]: r.calibration
                 for r in ctx._rirs.values()}
    with open(RAW_DIR / "calibration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for target in sorted(by_target):
            writer.writerow(by_target[target])


# ---------------------------------------------------------------------------
# Single-run debug mode
# ---------------------------------------------------------------------------

def run_single(cfg: dict, scenario_id: str, seed: int,
               systems: list[str]) -> None:
    scenario = cfg["scenarios"][scenario_id]
    ctx = BatchContext(cfg)
    sha = git_sha()
    for system in systems:
        spec = _make_spec("single", scenario_id, scenario_id, scenario, seed,
                          system,
                          record_traj=(system in ("nlms_f64", "nlms_q15")))
        row = process_row(ctx, spec, sha)
        conv = row.get("convergence_time_s")
        conv_str = f"{conv:.2f} s" if conv is not None else "not converged"
        erle_ss = row.get("erle_steady_state_db")
        erle_str = f"{erle_ss:5.1f} dB" if erle_ss is not None and \
            np.isfinite(erle_ss) else "  n/a"
        line = (f"{system:>8}: [{row['status']}] steady-state ERLE {erle_str}"
                f" | convergence {conv_str}")
        if row.get("misalignment_final_db") is not None:
            line += f" | misalign {row['misalignment_final_db']:.1f} dB"
        if row["fail_reason"]:
            line += f" | {row['fail_reason']}"
        print(line)
    _, rirs, _, scale, _ = ctx.signals(scenario, seed)
    print(f"RT60 target {rirs.rt60_target_s:.2f} s achieved "
          f"{rirs.rt60_achieved_echo_s:.3f} s | int16 scale {scale:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true",
                        help="run the full experiment matrix")
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+",
                        default=["none", "nlms_f64", "speex"],
                        choices=sorted(SYSTEM_RUNNERS))
    args = parser.parse_args()

    cfg = load_config()
    if args.batch:
        run_batch(cfg)
    elif args.scenario:
        if args.scenario not in cfg["scenarios"]:
            parser.error(f"unknown scenario {args.scenario!r}")
        run_single(cfg, args.scenario, args.seed, args.systems)
    else:
        parser.error("pass --batch or --scenario")


if __name__ == "__main__":
    main()
