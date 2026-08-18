"""Unit tests for §8.7: residual isolation and the masking model.

The reconstruction convention (hold = block-START state) is proven by
exactness at record_every=1: there the "held" state for sample n is the
state after sample n-1, which is precisely what the running filter used,
so reconstruction must match the actual output bit-for-bit (Q15) /
to float roundoff (f64). The masking model is tested against synthetic
cases with known answers; no constant is tuned here.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from aec_nlms import nlms
from aec_nlms_fixed import nlms_q15
from psychoacoustic import (
    audibility,
    bark_band_of_bin,
    hz_to_bark,
    masking_threshold,
    reconstruct_output_f64,
    reconstruct_output_q15,
    reconstruction_error_db,
    spreading_matrix,
    stft_band_power,
)

AUD = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config"
                          / "scenarios.yaml"))["audibility"]
FS = 16000


# ---------------------------------------------------------------------------
# Residual isolation
# ---------------------------------------------------------------------------

def _f64_run(seed=20, n=4000, L=32, record_every=1):
    rng = np.random.default_rng(seed)
    x = 0.2 * rng.standard_normal(n)
    h = rng.standard_normal(8) * np.exp(-0.4 * np.arange(8))
    d = np.convolve(x, h)[:n]
    out = nlms(x, d, L=L, mu=0.5, record_every=record_every)
    return x, d, out


def test_f64_reconstruction_exact_at_record_every_1():
    x, d, out = _f64_run(record_every=1)
    y_act = d - out["e"]
    y_rec = reconstruct_output_f64(x, out["w_traj"], 1)
    assert np.max(np.abs(y_rec - y_act)) < 1e-10


def test_f64_interpolation_exact_at_record_every_1():
    # With R=1, alpha is identically 0, so interp == hold == exact.
    x, d, out = _f64_run(record_every=1)
    y_act = d - out["e"]
    y_rec = reconstruct_output_f64(x, out["w_traj"], 1, interpolate=True)
    assert np.max(np.abs(y_rec - y_act)) < 1e-10


def test_f64_hold_and_interp_error_small_at_decimated_rate():
    x, d, out = _f64_run(n=8000, record_every=160)
    y_act = d - out["e"]
    valid = np.ones(len(x), bool)
    hold = reconstruct_output_f64(x, out["w_traj"], 160)
    interp = reconstruct_output_f64(x, out["w_traj"], 160,
                                    interpolate=True)
    e_hold = reconstruction_error_db(hold, y_act, valid)
    e_interp = reconstruction_error_db(interp, y_act, valid)
    assert e_hold < -10.0, e_hold
    # Interpolation must not be worse than hold on a converging run.
    assert e_interp <= e_hold + 1e-9, (e_interp, e_hold)


def test_q15_reconstruction_bit_exact_at_record_every_1():
    rng = np.random.default_rng(21)
    n = 3000
    x16 = np.round(4000 * rng.standard_normal(n)).astype(np.int16)
    d16 = np.round(0.5 * np.roll(x16, 5)).astype(np.int16)
    out = nlms_q15(x16, d16, L=16, mu_q15=16384, record_every=1,
                   shadow_float=False)
    y_act = d16.astype(np.int64) - out["e"].astype(np.int64)
    ok = np.ones(n, bool)
    ok[out["sat_positions"]["err"]] = False  # d-e != y where err clipped
    y_rec = reconstruct_output_q15(x16, out["w_traj"], 1)
    assert np.array_equal(y_rec.astype(np.int64)[ok], y_act[ok])


def test_q15_reconstruction_requires_int16():
    with pytest.raises(TypeError):
        reconstruct_output_q15(np.zeros(10), np.zeros((1, 4), np.int16), 1)


# ---------------------------------------------------------------------------
# Masking model
# ---------------------------------------------------------------------------

def test_bark_map_monotone_and_partitioning():
    band = bark_band_of_bin(FS, int(AUD["stft_nperseg"]), AUD)
    assert band[0] == 0
    assert np.all(np.diff(band) >= 0)
    assert np.all(np.diff(np.unique(band)) == 1)  # contiguous bands
    # 8 kHz Nyquist should land around 21 Bark for the Zwicker formula.
    assert 19 <= band[-1] <= 22
    assert abs(float(hz_to_bark(np.array([1000.0]), AUD)[0]) - 8.5) < 0.4


def test_spreading_matrix_decays_both_ways():
    S = spreading_matrix(10, AUD)
    assert np.all(np.diag(S) == 1.0)
    j = 5
    col = S[:, j]
    assert np.all(np.diff(col[:j]) > 0)   # rising toward the masker band
    assert np.all(np.diff(col[j:]) < 0)   # decaying above it
    # Upward spread (i > j) must be stronger than downward at the same
    # separation — the configured slopes say so.
    assert S[j + 2, j] > S[j - 2, j]


def test_residual_equal_to_masker_is_fully_audible():
    # Same signal as residual and masker: threshold sits offset dB below
    # the residual in every band, so every unit with energy above the
    # floor exceeds it and the mean excess is close to the offset.
    rng = np.random.default_rng(30)
    sig = 0.1 * rng.standard_normal(FS)
    valid = np.ones(50, bool)
    out = audibility(sig, sig, valid, 320, FS, AUD)
    assert out["fraction"] > 0.95
    assert abs(out["excess_db"] - float(AUD["masking_offset_db"])) < 2.0


def test_residual_well_below_masker_is_inaudible():
    rng = np.random.default_rng(31)
    masker = 0.1 * rng.standard_normal(FS)
    residual = 10 ** (-30 / 20) * masker  # 30 dB down, offset is 14
    valid = np.ones(50, bool)
    out = audibility(residual, masker, valid, 320, FS, AUD)
    assert out["fraction"] < 0.05


def test_silent_masker_uses_floor():
    # No masker: threshold = floor. A residual well above the floor is
    # ~fully audible; one well below is inaudible — and the silent-
    # masker cell must NOT report full audibility for a tiny residual.
    rng = np.random.default_rng(32)
    valid = np.ones(50, bool)
    loud = 0.1 * rng.standard_normal(FS)
    quiet = 1e-6 * rng.standard_normal(FS)
    silent = np.zeros(FS)
    assert audibility(loud, silent, valid, 320, FS, AUD)["fraction"] > 0.9
    assert audibility(quiet, silent, valid, 320, FS,
                      AUD)["fraction"] < 0.05


def test_tonal_masker_spreads_to_neighbouring_bands():
    # A tone masks its own band; a residual tone one Bark up (within the
    # upward spread) is less audible than one far away in frequency.
    t = np.arange(FS) / FS
    masker = 0.3 * np.sin(2 * np.pi * 1000 * t)
    near = 0.003 * np.sin(2 * np.pi * 1170 * t)   # ~1 Bark above 1 kHz
    far = 0.003 * np.sin(2 * np.pi * 6000 * t)    # many Bark away
    valid = np.ones(50, bool)
    a_near = audibility(near, masker, valid, 320, FS, AUD)["fraction"]
    a_far = audibility(far, masker, valid, 320, FS, AUD)["fraction"]
    assert a_far > a_near


def test_audibility_restricted_to_valid_frames():
    rng = np.random.default_rng(33)
    n = FS
    residual = np.zeros(n)
    residual[: n // 2] = 0.1 * rng.standard_normal(n // 2)  # loud 1st half
    masker = np.zeros(n)
    frame_len = 320
    n_seg = n // frame_len
    valid_first = np.zeros(n_seg, bool)
    valid_first[: n_seg // 2 - 2] = True
    valid_second = np.zeros(n_seg, bool)
    valid_second[n_seg // 2 + 2:] = True
    a1 = audibility(residual, masker, valid_first, frame_len, FS, AUD)
    a2 = audibility(residual, masker, valid_second, frame_len, FS, AUD)
    assert a1["fraction"] > 0.9   # loud half: audible over silence floor
    assert a2["fraction"] < 0.05  # silent half: nothing to hear


def test_no_valid_frames_returns_nan():
    out = audibility(np.zeros(FS), np.zeros(FS), np.zeros(50, bool),
                     320, FS, AUD)
    assert np.isnan(out["fraction"]) and np.isnan(out["excess_db"])
    assert out["n_units"] == 0
