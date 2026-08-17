"""Unit tests for the float64 NLMS implementation.

Convergence is asserted on coefficient misalignment against a known
synthetic echo path — not on ERLE, which conflates filter accuracy with
output energy.
"""

import numpy as np

from aec_nlms import nlms


def _misalignment_db(w: np.ndarray, h: np.ndarray) -> float:
    h_padded = np.zeros(len(w))
    h_padded[: len(h)] = h
    return 10 * np.log10(np.sum((w - h_padded) ** 2) / np.sum(h_padded**2))


def test_converges_to_known_path_on_white_noise():
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.standard_normal(n)
    h = rng.standard_normal(32) * np.exp(-0.15 * np.arange(32))
    d = np.convolve(x, h)[:n]

    out = nlms(x, d, L=64, mu=0.5)

    mis_db = _misalignment_db(out["w_final"], h)
    assert mis_db < -20.0, f"final misalignment {mis_db:.1f} dB"
    # Residual after convergence should be far below the echo.
    tail = slice(n // 2, None)
    assert np.mean(out["e"][tail] ** 2) < 1e-2 * np.mean(d[tail] ** 2)


def test_handles_delayed_path():
    # Echo path with a long leading-zero region (propagation delay), as the
    # simulated RIRs have. The filter must represent it, not skip it.
    rng = np.random.default_rng(4)
    n = 30000
    x = rng.standard_normal(n)
    h = np.zeros(120)
    h[87] = 0.8
    h[100] = 0.3
    d = np.convolve(x, h)[:n]

    out = nlms(x, d, L=200, mu=0.5)
    mis_db = _misalignment_db(out["w_final"], h)
    assert mis_db < -20.0, f"final misalignment {mis_db:.1f} dB"
    # The delay region must be (near) zero, not fitted with junk.
    assert np.max(np.abs(out["w_final"][:80])) < 0.05


def test_silence_leaves_filter_untouched():
    x = np.zeros(1000)
    d = np.ones(1000) * 0.1
    out = nlms(x, d, L=32, mu=0.5)
    assert np.allclose(out["e"], d)  # nothing to cancel from
    assert np.allclose(out["w_final"], 0.0)  # no update without input power


def _nlms_naive_pure_python(x, d, L, mu, delta):
    """Per-sample reference implementation: pure Python scalars, explicit
    tap buffer, no numpy in the loop. The ground truth for exactness."""
    n = len(x)
    w = [0.0] * L        # w[0] pairs with the newest sample
    buf = [0.0] * L      # buf[0] = x(i), buf[1] = x(i-1), ...
    e = [0.0] * n
    for i in range(n):
        buf = [float(x[i])] + buf[:-1]
        y = sum(wk * bk for wk, bk in zip(w, buf))
        err = float(d[i]) - y
        e[i] = err
        norm = sum(b * b for b in buf)
        g = mu * err / (norm + delta)
        w = [wk + g * bk for wk, bk in zip(w, buf)]
    return np.array(e)


def test_fast_implementation_is_samplewise_exact():
    # The fast implementation must be the same sample-wise algorithm as a
    # naive per-sample loop — speed does not license changing the system
    # under test. Serial dependency: w(n+1) depends on e(n).
    rng = np.random.default_rng(6)
    n = 2000
    x = rng.standard_normal(n)
    h = rng.standard_normal(16) * np.exp(-0.2 * np.arange(16))
    d = np.convolve(x, h)[:n] + 0.01 * rng.standard_normal(n)

    fast = nlms(x, d, L=32, mu=0.5, delta=1e-6)
    ref = _nlms_naive_pure_python(x, d, L=32, mu=0.5, delta=1e-6)

    max_diff = np.max(np.abs(fast["e"] - ref))
    assert max_diff < 1e-12, f"per-sample deviation {max_diff:.2e}"


def test_sliding_power_survives_full_length_run():
    # 15 s at 16 kHz with the baseline 3200-tap filter: the in-loop periodic
    # exact recomputation (POWER_CHECK_PERIOD) raises if the O(1) power
    # estimate ever drifts beyond POWER_CHECK_REL_TOL.
    rng = np.random.default_rng(7)
    n = 240000
    x = 0.05 * rng.standard_normal(n)
    d = 0.5 * np.roll(x, 87)
    nlms(x, d, L=3200, mu=0.5)  # must not raise


def test_trajectory_shape_and_consistency():
    rng = np.random.default_rng(5)
    n = 1000
    x = rng.standard_normal(n)
    d = 0.5 * np.roll(x, 3)
    out = nlms(x, d, L=8, mu=0.5, record_every=100)
    assert out["w_traj"].shape == (10, 8)
    assert np.allclose(out["w_traj"][-1], out["w_final"])
    out_none = nlms(x, d, L=8, mu=0.5)
    assert out_none["w_traj"] is None
    assert np.allclose(out_none["e"], out["e"])  # recording must not alter it
