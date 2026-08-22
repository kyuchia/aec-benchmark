"""Evaluation metrics: ERLE, convergence time, distortion, misalignment.

All functions take numpy arrays plus explicit parameters and return
plain values — no file I/O, no config loading. Frame-based metrics use
the same non-overlapping frame grid as the segmentation.

Choices that matter for interpretation (all stated in the report):
- ERLE is computed only over ERLE-valid frames (far-active and not
  near-active) and smoothed with an EMA over the valid-frame sequence.
- The steady-state ERLE sanity assertion (> erle_sanity_max_db fails)
  applies to the steady-state statistic only; instantaneous short-time
  ERLE can legitimately spike far higher in low-energy frames.
- Segmental SNR references s (the reverberant near-end at the mic), not
  s_clean: the AEC is not being asked to dereverberate.
"""

from __future__ import annotations

import numpy as np

from segment import frame_power


# ---------------------------------------------------------------------------
# ERLE and convergence time
# ---------------------------------------------------------------------------

def erle_curve(d: np.ndarray, e: np.ndarray, erle_valid: np.ndarray,
               frame_len: int, ema_alpha: float,
               steady_state_last_fraction: float,
               sanity_max_db: float) -> dict:
    """Short-time ERLE over valid frames, EMA-smoothed, plus steady state.

    Returns {'erle_db': (n_frames,) raw, NaN on invalid frames,
             'erle_smoothed_db': same shape, EMA over valid frames only,
             'steady_state_db': median of the final fraction of valid
             smoothed values, 'n_valid_frames': int}.

    Raises AssertionError if the steady-state value exceeds
    sanity_max_db — that almost certainly means the AEC was given the
    echo itself (d_echo) instead of the loudspeaker signal x as its
    reference, and every downstream number would be meaningless.
    """
    p_d = frame_power(d, frame_len)
    p_e = frame_power(e, frame_len)
    n_frames = len(p_d)
    if len(erle_valid) != n_frames:
        raise ValueError(
            f"segmentation has {len(erle_valid)} frames, signal {n_frames}")

    raw = np.full(n_frames, np.nan)
    smoothed = np.full(n_frames, np.nan)
    acc = None
    with np.errstate(divide="ignore"):
        for i in np.flatnonzero(erle_valid):
            val = 10.0 * np.log10(p_d[i] / p_e[i]) if p_e[i] > 0 else np.inf
            raw[i] = val
            acc = val if acc is None else (
                ema_alpha * acc + (1.0 - ema_alpha) * val)
            smoothed[i] = acc

    valid_smoothed = smoothed[erle_valid]
    n_valid = len(valid_smoothed)
    if n_valid == 0:
        steady_state = np.nan
    else:
        tail = valid_smoothed[int(n_valid * (1.0 - steady_state_last_fraction)):]
        steady_state = float(np.median(tail))
        if steady_state > sanity_max_db:
            raise AssertionError(
                f"steady-state ERLE {steady_state:.1f} dB exceeds "
                f"{sanity_max_db} dB. This almost certainly means the "
                "reference input is wrong (d_echo passed instead of x — "
                "estimating the echo path is the AEC's job). Check the "
                "signal wiring before trusting any result."
            )

    return {
        "erle_db": raw,
        "erle_smoothed_db": smoothed,
        "steady_state_db": steady_state,
        "n_valid_frames": int(n_valid),
    }


def convergence_time_s(erle_smoothed_db: np.ndarray, erle_valid: np.ndarray,
                       steady_state_db: float, frame_times_s: np.ndarray,
                       convergence_fraction: float) -> float:
    """Time from the first valid frame until smoothed ERLE first reaches
    convergence_fraction * steady state. NaN if never reached (the caller
    must flag it as non-converged, never substitute a value)."""
    valid_idx = np.flatnonzero(erle_valid)
    if len(valid_idx) == 0 or not np.isfinite(steady_state_db):
        return float("nan")
    target = convergence_fraction * steady_state_db
    onset_t = frame_times_s[valid_idx[0]]
    for i in valid_idx:
        if erle_smoothed_db[i] >= target:
            return float(frame_times_s[i] - onset_t)
    return float("nan")


# ---------------------------------------------------------------------------
# Near-end distortion during double-talk
# ---------------------------------------------------------------------------

def segmental_snr_db(ref: np.ndarray, est: np.ndarray,
                     frames_mask: np.ndarray, frame_len: int) -> dict:
    """Mean per-frame SNR of est against ref over the masked frames.

    Frames where ref has zero power are excluded (and counted): SNR is
    undefined there. Frames with exactly zero error (bit-exact
    passthrough) are likewise excluded and counted — they carry no finite
    SNR; if every usable frame is error-free the result is +inf, not a
    floor-dependent large number. No per-frame clamping is applied; the
    per-frame values are returned so outliers stay visible.
    """
    p_ref = frame_power(ref, frame_len)
    p_err = frame_power(ref - est, frame_len)
    idx = np.flatnonzero(frames_mask[: len(p_ref)])
    nonzero_ref = idx[p_ref[idx] > 0.0]
    usable = nonzero_ref[p_err[nonzero_ref] > 0.0]
    per_frame = 10.0 * np.log10(p_ref[usable] / p_err[usable])
    if len(usable):
        segsnr = float(np.mean(per_frame))
    elif len(nonzero_ref):
        segsnr = float("inf")   # every frame reproduced exactly
    else:
        segsnr = float("nan")
    return {
        "segsnr_db": segsnr,
        "per_frame_db": per_frame,
        "n_frames": int(len(usable)),
        "n_excluded_zero_ref": int(len(idx) - len(nonzero_ref)),
        "n_excluded_zero_error": int(len(nonzero_ref) - len(usable)),
    }


def log_spectral_distance_db(ref: np.ndarray, est: np.ndarray,
                             frames_mask: np.ndarray, frame_len: int,
                             floor_db: float) -> dict:
    """LSD per masked frame: rms over frequency of the dB spectral
    difference between ref and est (rfft per frame, power floored)."""
    n_frames = min(len(ref), len(est)) // frame_len
    idx = np.flatnonzero(frames_mask[:n_frames])
    if len(idx) == 0:
        return {"lsd_db": float("nan"), "per_frame_db": np.array([]),
                "n_frames": 0}
    floor = 10.0 ** (floor_db / 10.0)
    per_frame = np.empty(len(idx))
    for k, i in enumerate(idx):
        sl = slice(i * frame_len, (i + 1) * frame_len)
        p_ref = np.maximum(np.abs(np.fft.rfft(ref[sl])) ** 2, floor)
        p_est = np.maximum(np.abs(np.fft.rfft(est[sl])) ** 2, floor)
        diff_db = 10.0 * np.log10(p_ref / p_est)
        per_frame[k] = np.sqrt(np.mean(diff_db**2))
    return {
        "lsd_db": float(np.mean(per_frame)),
        "per_frame_db": per_frame,
        "n_frames": int(len(idx)),
    }


def speech_quality_scores(ref: np.ndarray, est: np.ndarray,
                          sample_rate: int) -> dict:
    """STOI and PESQ (wideband) of est against ref over the given span.

    Import errors or metric failures are reported, not masked — a metric
    that could not be computed must show up as missing, not as silence.
    """
    out: dict = {}
    try:
        from pystoi import stoi
        out["stoi"] = float(stoi(ref, est, sample_rate, extended=False))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        out["stoi"] = None
        out["stoi_error"] = repr(exc)
    try:
        from pesq import pesq
        out["pesq_wb"] = float(pesq(sample_rate, ref, est, "wb"))
    except Exception as exc:  # noqa: BLE001
        out["pesq_wb"] = None
        out["pesq_error"] = repr(exc)
    return out


# ---------------------------------------------------------------------------
# Coefficient misalignment
# ---------------------------------------------------------------------------

def misalignment_db(w: np.ndarray, h: np.ndarray) -> float:
    """10 log10(||w - h||^2 / ||h||^2), h truncated/zero-padded to len(w)."""
    h_fit = np.zeros(len(w))
    n = min(len(w), len(h))
    h_fit[:n] = h[:n]
    denom = float(np.sum(h_fit**2))
    if denom == 0.0:
        raise ValueError("echo path is all zero after truncation")
    return float(10.0 * np.log10(np.sum((w - h_fit) ** 2) / denom))


def misalignment_curve_db(w_traj: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Misalignment per recorded coefficient snapshot."""
    return np.array([misalignment_db(w, h) for w in w_traj])


# ---------------------------------------------------------------------------
# Computational cost — derived analytically, never measured
# ---------------------------------------------------------------------------

def nlms_mac_per_sample(L: int) -> float:
    """Sample-wise NLMS: L MACs for the filter output plus L for the
    coefficient update per sample; the sliding-window power and the
    normalised gain are O(1) per sample and excluded. Identical for the
    float and Q15 paths (a MAC is a MAC; only its width differs)."""
    return float(2 * L)


def mdf_mac_per_sample(frame_size: int, L: int) -> float:
    """Partitioned block-frequency-domain (MDF/AUMDF) canceller,
    K = ceil(L/N) partitions, FFT length M = 2N, per N-sample frame:

      - 5 real FFTs of length 2N (input block, filter output, error,
        and the amortised AUMDF gradient-constraint pair), each costed
        at 5*N*log2(2N) real MACs — half the classic 5*M*log2(M)
        real-operation count of a complex radix-2-class FFT at M = 2N;
      - filtering + update: one complex multiply-accumulate per bin per
        partition for each, 2 * K * (N+1) complex MACs = 8 * K * (N+1)
        real MACs.

    Total per frame: 25*N*log2(2N) + 8*K*(N+1); divided by N for the
    per-sample figure."""
    n = int(frame_size)
    k = -(-int(L) // n)  # ceil
    per_frame = 25.0 * n * np.log2(2 * n) + 8.0 * k * (n + 1)
    return per_frame / n


def nlms_f64_state_bytes(L: int) -> int:
    """float64 coefficients (8L) + L-sample float64 input window (8L);
    the sliding power sum and scalars are O(1)."""
    return 16 * L


def nlms_q15_state_bytes(L: int) -> int:
    """int16 coefficients (2L) + int16 input window (2L) + the int64
    exact window-power accumulator. This is the algorithmic state; the
    reference implementation additionally widens the coefficient array
    to int64 in-loop for analysis convenience, which is not counted."""
    return 4 * L + 8


def mdf_state_bytes(frame_size: int, L: int) -> int:
    """SpeexDSP AUMDF, float build: dominant arrays are the adaptive
    weights (K*M floats), the foreground filter of the two-filter
    structure (K*M), and the input-spectrum history ((K+1)*M), M = 2N —
    (3K+1)*M*4 bytes, plus O(M) working buffers not counted. Derived
    from the AUMDF structure, not measured."""
    n = int(frame_size)
    k = -(-int(L) // n)
    return (3 * k + 1) * 2 * n * 4
