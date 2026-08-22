"""Perceptual audibility of residual echo.

Two halves, both belonging to the same analysis:

1. Residual-echo isolation. For the linear NLMS systems this is exact
   given the recorded coefficient trajectory: apply the time-varying
   filter w(n) to x and subtract from d_echo. Between snapshots
   (record_every samples apart) the coefficients are HELD at the
   block-start state — the convention that is causally exact when
   record_every=1 (y(n) is computed from the state after sample n-1),
   which is how the reconstruction is unit-tested bit-exactly. A linear
   interpolation between snapshots is available for the float path as a
   secondary variant; interpolating int16 states is not Q15 arithmetic,
   so the fixed-point path is hold-only. The Q15 reconstruction runs the
   SAME Q15 arithmetic as aec_nlms_fixed (int64 accumulation, magnitude
   truncation by 15, saturation) on the recorded int16 trajectory — a
   float reconstruction of the fixed-point filter would reintroduce
   exactly the arithmetic difference M7 measured.

   QC oracle: the run's actual filter output is recoverable exactly as
   y_actual = d - e (float path) / d16 - e16 (Q15 path, except samples
   where the error narrowing saturated — those are recorded and
   excluded). Reconstruction error is reported in dB relative to the
   actual output's energy over ERLE-valid samples.

2. Simplified simultaneous-masking model: STFT residual and masker,
   map power to Bark bands (Zwicker–Terhardt arctan approximation),
   spread across bands with a two-slope linear-domain excitation sum,
   subtract a fixed offset to obtain the threshold, and floor the
   threshold at a fixed constant (the absolute-threshold proxy — without
   it, no-noise far-single cells would report ~100% audibility by
   construction, since their masker is silence). All constants live in
   config/scenarios.yaml under `audibility:`.

   Outputs: fraction of time–frequency (band × frame) units where the
   residual exceeds the threshold, and the mean excess above threshold
   in dB — computed over the same far-single (ERLE-valid) frames the
   ERLE metric uses, so the two are comparable row by row.

This is deliberately NOT an off-the-shelf PEAQ/psychoacoustics package:
the analysis asks a relative question (which system's residual is more
audible under the same masker), and a fixed-offset simplified model
biases all systems identically, so comparative conclusions survive the
simplification. Absolute audibility in phons/sones is out of scope.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

from aec_nlms_fixed import Q15_MAX, Q15_MIN


# ---------------------------------------------------------------------------
# Residual isolation: time-varying filter reconstruction from w_traj
# ---------------------------------------------------------------------------

def _padded_windows(x: np.ndarray, L: int) -> np.ndarray:
    """Row n = [x(n-L+1), ..., x(n)] — oldest first, zero history."""
    xp = np.concatenate([np.zeros(L - 1, x.dtype), x])
    return sliding_window_view(xp, L)


def reconstruct_output_f64(x: np.ndarray, w_traj: np.ndarray,
                           record_every: int,
                           interpolate: bool = False) -> np.ndarray:
    """y(n) from snapshots. w_traj rows are newest-first (as returned by
    the AECs); w_traj[k] is the state after sample (k+1)*record_every.
    Hold: block k (samples [k*R, (k+1)*R)) uses the state entering it —
    w_traj[k-1], zeros for k=0. Interpolate: linear blend from the
    block-start to the block-end state across the block."""
    n = len(x)
    L = w_traj.shape[1]
    win = _padded_windows(np.asarray(x, np.float64), L)
    traj = w_traj[:, ::-1]  # oldest-first, matching the window rows
    y = np.empty(n)
    zeros = np.zeros(L)
    for k in range(int(np.ceil(n / record_every))):
        n0, n1 = k * record_every, min((k + 1) * record_every, n)
        w_start = traj[k - 1] if k > 0 else zeros
        y_start = win[n0:n1] @ w_start
        if interpolate and k < len(traj):
            w_end = traj[k]
            y_end = win[n0:n1] @ w_end
            # y(n) uses the state entering sample n (after n-1's update):
            # at the block's first sample none of its updates have been
            # applied yet, so alpha starts at 0 (pure block-start state,
            # matching the hold path exactly there).
            alpha = (np.arange(n0, n1) - n0) / record_every
            y[n0:n1] = (1.0 - alpha) * y_start + alpha * y_end
        else:
            y[n0:n1] = y_start
    return y


def reconstruct_output_q15(x16: np.ndarray, w_traj: np.ndarray,
                           record_every: int) -> np.ndarray:
    """Q15 hold reconstruction: same arithmetic as aec_nlms_fixed's
    output path (int64 accumulate, magnitude truncation by 15,
    saturate). w_traj is the recorded int16 trajectory, newest-first
    rows. Returns int16 y."""
    if x16.dtype != np.int16 or w_traj.dtype != np.int16:
        raise TypeError("Q15 reconstruction needs int16 x16 and w_traj")
    n = len(x16)
    L = w_traj.shape[1]
    win = _padded_windows(x16.astype(np.int64), L)
    traj = w_traj[:, ::-1].astype(np.int64)
    y = np.empty(n, np.int16)
    zeros = np.zeros(L, np.int64)
    for k in range(int(np.ceil(n / record_every))):
        n0, n1 = k * record_every, min((k + 1) * record_every, n)
        w_start = traj[k - 1] if k > 0 else zeros
        acc = win[n0:n1] @ w_start
        trunc = np.where(acc >= 0, acc >> 15, -((-acc) >> 15))
        y[n0:n1] = np.clip(trunc, Q15_MIN, Q15_MAX)
    return y


def reconstruction_error_db(y_rec: np.ndarray, y_act: np.ndarray,
                            valid_samples: np.ndarray) -> float:
    """10 log10 of reconstruction-error energy over actual-output energy,
    on the given sample mask. +inf-safe: silent actual output -> NaN."""
    r = np.asarray(y_rec, np.float64)[valid_samples]
    a = np.asarray(y_act, np.float64)[valid_samples]
    denom = float(a @ a)
    if denom <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(max(float((r - a) @ (r - a)), 1e-300)
                                 / denom))


# ---------------------------------------------------------------------------
# Masking model
# ---------------------------------------------------------------------------

def hz_to_bark(f_hz: np.ndarray, aud_cfg: dict) -> np.ndarray:
    """Zwicker–Terhardt arctan approximation; constants from config."""
    a = float(aud_cfg["bark_a"])
    b = float(aud_cfg["bark_b"])
    c = float(aud_cfg["bark_c"])
    d = float(aud_cfg["bark_d_hz"])
    f = np.asarray(f_hz, np.float64)
    return a * np.arctan(b * f) + c * np.arctan((f / d) ** 2)


def bark_band_of_bin(sample_rate: int, nperseg: int,
                     aud_cfg: dict) -> np.ndarray:
    """Integer Bark band index for each rfft bin (floor of the bin
    centre's Bark value); bands partition the bins contiguously."""
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    band = np.floor(hz_to_bark(freqs, aud_cfg)).astype(int)
    assert np.all(np.diff(band) >= 0), "Bark band map must be monotone"
    return band


def stft_band_power(sig: np.ndarray, sample_rate: int,
                    aud_cfg: dict) -> np.ndarray:
    """(n_bands, n_frames) power per Bark band; frame k covers samples
    [k*hop, k*hop + nperseg). No padding: only complete frames."""
    nperseg = int(aud_cfg["stft_nperseg"])
    hop = int(aud_cfg["stft_hop"])
    w = get_window(aud_cfg["stft_window"], nperseg, fftbins=True)
    n_frames = 1 + (len(sig) - nperseg) // hop
    frames = sliding_window_view(np.asarray(sig, np.float64),
                                 nperseg)[::hop][:n_frames]
    spec = np.fft.rfft(frames * w, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) / np.sum(w ** 2)
    band = bark_band_of_bin(sample_rate, nperseg, aud_cfg)
    n_bands = int(band.max()) + 1
    out = np.zeros((n_bands, n_frames))
    for bb in range(n_bands):
        out[bb] = power[:, band == bb].sum(axis=1)
    return out


def spreading_matrix(n_bands: int, aud_cfg: dict) -> np.ndarray:
    """S[i, j]: linear-domain gain from masker band j into band i, from
    the two-slope spreading function."""
    lo = float(aud_cfg["spreading_lower_db_per_bark"])
    up = float(aud_cfg["spreading_upper_db_per_bark"])
    i = np.arange(n_bands)[:, None]
    j = np.arange(n_bands)[None, :]
    att_db = np.where(i >= j, up * (i - j), lo * (j - i))
    return 10.0 ** (-att_db / 10.0)


def masking_threshold(masker_bands: np.ndarray,
                      aud_cfg: dict) -> np.ndarray:
    """Threshold per (band, frame): spread the masker excitation across
    bands, subtract the fixed offset, floor at the configured constant."""
    S = spreading_matrix(masker_bands.shape[0], aud_cfg)
    spread = S @ masker_bands
    thr = spread * 10.0 ** (-float(aud_cfg["masking_offset_db"]) / 10.0)
    floor = 10.0 ** (float(aud_cfg["threshold_floor_db"]) / 10.0)
    return np.maximum(thr, floor)


def audibility(residual: np.ndarray, masker: np.ndarray,
               erle_valid: np.ndarray, seg_frame_len: int,
               sample_rate: int, aud_cfg: dict) -> dict:
    """Audibility outputs over ERLE-valid frames.

    Returns fraction of TF units above threshold, mean excess (dB) over
    those units, unit count, and the per-frame exceedance-fraction
    curve (NaN on invalid frames)."""
    res_b = stft_band_power(residual, sample_rate, aud_cfg)
    mask_b = stft_band_power(masker, sample_rate, aud_cfg)
    thr = masking_threshold(mask_b, aud_cfg)

    n_frames = res_b.shape[1]
    hop = int(aud_cfg["stft_hop"])
    nperseg = int(aud_cfg["stft_nperseg"])
    centers = np.arange(n_frames) * hop + nperseg // 2
    seg_idx = np.minimum(centers // seg_frame_len, len(erle_valid) - 1)
    valid = erle_valid[seg_idx]

    frame_frac = np.full(n_frames, np.nan)
    exceed = res_b > thr
    frame_frac[:] = exceed.mean(axis=0)
    frame_frac[~valid] = np.nan

    if not np.any(valid):
        return {"fraction": float("nan"), "excess_db": float("nan"),
                "n_units": 0, "frame_fraction": frame_frac,
                "valid_stft_frames": valid}
    ev = exceed[:, valid]
    n_units = int(ev.size)
    frac = float(ev.mean())
    if ev.any():
        excess = 10.0 * np.log10(res_b[:, valid][ev] / thr[:, valid][ev])
        excess_db = float(np.mean(excess))
    else:
        excess_db = float("nan")
    return {"fraction": frac, "excess_db": excess_db, "n_units": n_units,
            "frame_fraction": frame_frac, "valid_stft_frames": valid}
