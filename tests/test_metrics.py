"""Unit tests for metrics against synthetic inputs with known answers."""

import numpy as np
import pytest

import metrics

FRAME = 320


def test_erle_constant_ratio_is_exact():
    # e is exactly 20 dB below d in every frame -> ERLE = 20 dB everywhere,
    # raw, smoothed, and steady-state alike.
    rng = np.random.default_rng(0)
    d = rng.standard_normal(100 * FRAME)
    e = d * 10.0 ** (-20.0 / 20.0)
    valid = np.ones(100, dtype=bool)
    out = metrics.erle_curve(d, e, valid, FRAME, ema_alpha=0.9,
                             steady_state_last_fraction=0.3,
                             sanity_max_db=60.0)
    np.testing.assert_allclose(out["erle_db"][valid], 20.0, atol=1e-9)
    np.testing.assert_allclose(out["steady_state_db"], 20.0, atol=1e-9)
    assert out["n_valid_frames"] == 100


def test_erle_invalid_frames_are_excluded():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(10 * FRAME)
    e = d * 0.1
    valid = np.zeros(10, dtype=bool)
    valid[3:7] = True
    out = metrics.erle_curve(d, e, valid, FRAME, 0.9, 0.3, 60.0)
    assert np.all(np.isnan(out["erle_db"][~valid]))
    assert np.all(np.isfinite(out["erle_db"][valid]))
    assert out["n_valid_frames"] == 4


def test_erle_sanity_assertion_fires_on_steady_state_only():
    rng = np.random.default_rng(2)
    d = rng.standard_normal(100 * FRAME)
    # Implausibly high steady state (the wrong-reference signature).
    e = d * 10.0 ** (-70.0 / 20.0)
    valid = np.ones(100, dtype=bool)
    with pytest.raises(AssertionError, match="reference"):
        metrics.erle_curve(d, e, valid, FRAME, 0.9, 0.3, 60.0)

    # A brief 70 dB spike within an otherwise 20 dB run must NOT fire:
    # the assertion applies to the steady-state statistic, not per-window.
    e2 = d * 10.0 ** (-20.0 / 20.0)
    spike = slice(10 * FRAME, 11 * FRAME)
    e2[spike] = d[spike] * 10.0 ** (-70.0 / 20.0)
    out = metrics.erle_curve(d, e2, valid, FRAME, 0.9, 0.3, 60.0)
    assert out["steady_state_db"] < 25.0


def test_convergence_time_known_ramp():
    # Smoothed ERLE ramps 0..30 dB over frames 0..59 then holds 30 dB.
    n = 100
    smoothed = np.concatenate([np.linspace(0, 30, 60), np.full(40, 30.0)])
    valid = np.ones(n, dtype=bool)
    times = (np.arange(n) + 0.5) * 0.02
    # Steady state 30 dB, 90% = 27 dB, first reached at frame 54.
    t = metrics.convergence_time_s(smoothed, valid, 30.0, times, 0.9)
    assert abs(t - (times[54] - times[0])) < 1e-12


def test_convergence_never_reached_is_nan():
    n = 50
    smoothed = np.full(n, 10.0)
    valid = np.ones(n, dtype=bool)
    times = (np.arange(n) + 0.5) * 0.02
    t = metrics.convergence_time_s(smoothed, valid, 20.0, times, 0.9)
    assert np.isnan(t)


def test_segmental_snr_known_ratio():
    # est = ref + err with err exactly 20 dB below ref per frame.
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(20 * FRAME)
    err_shape = rng.standard_normal(20 * FRAME)
    # scale error per frame to exactly -20 dB relative to ref
    err = np.empty_like(ref)
    for i in range(20):
        sl = slice(i * FRAME, (i + 1) * FRAME)
        g = np.sqrt(np.mean(ref[sl] ** 2) / np.mean(err_shape[sl] ** 2))
        err[sl] = err_shape[sl] * g * 0.1
    out = metrics.segmental_snr_db(ref, ref + err, np.ones(20, dtype=bool),
                                   FRAME)
    np.testing.assert_allclose(out["segsnr_db"], 20.0, atol=1e-9)
    assert out["n_frames"] == 20


def test_segmental_snr_excludes_zero_reference_frames():
    ref = np.zeros(10 * FRAME)
    ref[: 5 * FRAME] = 1.0
    est = ref + 0.1
    out = metrics.segmental_snr_db(ref, est, np.ones(10, dtype=bool), FRAME)
    assert out["n_frames"] == 5
    assert out["n_excluded_zero_ref"] == 5


def test_lsd_flat_gain_is_exact():
    # est = 0.5 * ref: power ratio 4 -> |10 log10 4| ~= 6.0206 dB in every
    # bin, so the rms over frequency equals it exactly.
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(10 * FRAME)
    est = 0.5 * ref
    out = metrics.log_spectral_distance_db(ref, est, np.ones(10, dtype=bool),
                                           FRAME, floor_db=-120.0)
    np.testing.assert_allclose(out["lsd_db"], 10 * np.log10(4.0), atol=1e-6)


def test_misalignment_known_error():
    # ||w - h||^2 / ||h||^2 = 0.01 exactly -> -20 dB
    h = np.array([1.0, 0.0])
    w = np.array([1.0, 0.1])
    np.testing.assert_allclose(metrics.misalignment_db(w, h), -20.0,
                               atol=1e-12)
    # Exact match (h zero-padded to len(w)) -> -inf.
    h2 = np.array([1.0, 0.5, 0.25])
    w2 = np.array([1.0, 0.5, 0.25, 0.0, 0.0])
    with np.errstate(divide="ignore"):
        assert metrics.misalignment_db(w2, h2) == -np.inf


def test_misalignment_truncates_long_path():
    # h longer than w: only the first len(w) taps are compared, so a w that
    # matches the truncated support exactly scores -inf.
    h = np.array([1.0, 0.0, 0.0, 5.0])
    w = np.array([1.0, 0.0])
    with np.errstate(divide="ignore"):
        assert metrics.misalignment_db(w, h) == -np.inf


def test_misalignment_curve_matches_scalar():
    h = np.array([1.0, 0.0])
    traj = np.array([[0.5, 0.0], [0.9, 0.0], [1.0, 0.1]])
    curve = metrics.misalignment_curve_db(traj, h)
    expected = [metrics.misalignment_db(w, h) for w in traj]
    np.testing.assert_allclose(curve, expected)
