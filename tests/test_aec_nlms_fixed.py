"""Unit tests for the Q15 fixed-point NLMS implementation.

The correctness reference is _nlms_q15_naive below: the spec §7.3
arithmetic executed literally with pure-Python unbounded integers, an
explicit tap buffer, an explicit saturation function, and per-sample
recomputation of the window power from the buffer. Python ints cannot
wrap, so bit-exact equality against this reference proves both that the
numpy path never wraps silently and that its exact sliding-window power
matches full recomputation.

Convergence is asserted on coefficient misalignment against a known
synthetic echo path, bracketed by the expected Q15 noise floor: pure
coefficient rounding alone contributes about
    10 log10( L * (2^-15)^2 / 12 / ||h||^2 )
(~ -80 dB for the test geometry below), and adaptation/truncation noise
sits well above that — so the assertion window is (-80, -25) dB,
nowhere near the float path's ~ -309 dB.
"""

import numpy as np

from aec_nlms import nlms
from aec_nlms_fixed import (
    DELTA_Q30_DEFAULT,
    _sat16_scalar,
    _sat16_vec,
    nlms_q15,
    quantise_coeffs,
)


def _misalignment_db(w: np.ndarray, h: np.ndarray) -> float:
    h_padded = np.zeros(len(w))
    h_padded[: len(h)] = h
    return 10 * np.log10(np.sum((w - h_padded) ** 2) / np.sum(h_padded**2))


def _to_q15(sig: np.ndarray) -> np.ndarray:
    scaled = np.round(sig * 32768.0)
    assert np.all(np.abs(scaled) <= 32767), "test signal would clip"
    return scaled.astype(np.int16)


def _echo_signals(seed: int, n: int, x_rms: float = 0.15) -> tuple:
    """(x16, d16, h): white far-end through a short known echo path."""
    rng = np.random.default_rng(seed)
    x = np.clip(x_rms * rng.standard_normal(n), -0.99, 0.99)
    h = np.array([0.5, -0.35, 0.25, -0.15, 0.1, -0.06, 0.04, -0.02])
    d = np.convolve(x, h)[:n]
    return _to_q15(x), _to_q15(d), h


# ---------------------------------------------------------------------------
# Naive reference: §7.3 arithmetic, pure-Python unbounded ints
# ---------------------------------------------------------------------------

def _sat(v: int) -> tuple[int, bool]:
    if v > 32767:
        return 32767, True
    if v < -32768:
        return -32768, True
    return v, False


def _nlms_q15_naive(x16, d16, L, mu, delta, bits):
    n = len(x16)
    shift = 15 - bits
    w = [0] * L    # w[0] pairs with the newest sample
    buf = [0] * L  # buf[0] = x(i), buf[1] = x(i-1), ...
    e = [0] * n
    stall = []
    tap_stalls = [0] * n
    sat_counts = {"y": 0, "err": 0, "gain": 0, "coeff": 0}
    for i in range(n):
        buf = [int(x16[i])] + buf[:-1]
        acc = sum(wk * bk for wk, bk in zip(w, buf))
        y, s_ = _sat(acc >> 15)
        sat_counts["y"] += s_
        err, s_ = _sat(int(d16[i]) - y)
        sat_counts["err"] += s_
        e[i] = err
        p = sum(b * b for b in buf)  # full recomputation every sample
        if err != 0 and p > 0:
            p64 = p + delta
            s = p64.bit_length() - 15
            m = p64 >> s if s >= 0 else p64 << -s
            r = min((1 << 29) // m, 32767)
            g, s_ = _sat((mu * err * r) >> (14 + s))  # g = 2^15*mu*err/p64
            sat_counts["gain"] += s_
            if g != 0:
                new_w = []
                clipped = changed = False
                for k in range(L):
                    cand, s_ = _sat(w[k] + ((g * buf[k]) >> 15))
                    clipped |= s_
                    if shift:
                        cand = (cand >> shift) << shift
                    changed |= cand != w[k]
                    if buf[k] != 0 and cand == w[k]:
                        tap_stalls[i] += 1
                    new_w.append(cand)
                w = new_w
                sat_counts["coeff"] += clipped
                if not changed:
                    stall.append(i)
            else:
                tap_stalls[i] = sum(1 for b in buf if b != 0)
                stall.append(i)
    return {
        "e": np.array(e, np.int16),
        "w_final": np.array(w, np.int64),  # newest-first already
        "stall_positions": np.array(stall, np.int64),
        "tap_stalls_per_sample": np.array(tap_stalls, np.int32),
        "sat_counts": sat_counts,
    }


def _assert_bit_exact(x16, d16, L, mu, bits):
    fast = nlms_q15(x16, d16, L=L, mu_q15=mu, coeff_bits=bits,
                    shadow_float=False)
    ref = _nlms_q15_naive(x16, d16, L=L, mu=mu,
                          delta=DELTA_Q30_DEFAULT, bits=bits)
    assert np.array_equal(fast["e"], ref["e"])
    assert np.array_equal(fast["w_final"].astype(np.int64), ref["w_final"])
    assert np.array_equal(fast["stall_positions"], ref["stall_positions"])
    assert np.array_equal(fast["tap_stalls_per_sample"],
                          ref["tap_stalls_per_sample"])
    assert fast["sat_counts"] == ref["sat_counts"]


def test_bit_exact_vs_naive_reference():
    x16, d16, _ = _echo_signals(seed=10, n=2000)
    _assert_bit_exact(x16, d16, L=16, mu=16384, bits=15)


def test_bit_exact_vs_naive_reference_masked():
    # Same proof with the word-length mask active in both implementations.
    x16, d16, _ = _echo_signals(seed=11, n=2000)
    _assert_bit_exact(x16, d16, L=16, mu=16384, bits=9)


def test_bit_exact_under_heavy_saturation():
    # Rail-heavy signal: d demands a -1.2 echo gain, below the Q15
    # coefficient range, so the 'coeff' site clips constantly. The naive
    # reference cannot wrap, so equality here is the strongest
    # wraparound proof.
    rng = np.random.default_rng(12)
    n = 1500
    x16 = (rng.choice([-1, 1], n) * 20000).astype(np.int16)
    d16 = (-1.2 * x16).astype(np.float64)
    d16 = np.clip(np.round(d16), -32768, 32767).astype(np.int16)
    fast = nlms_q15(x16, d16, L=8, mu_q15=16384, shadow_float=False)
    assert fast["sat_counts"]["coeff"] > 0
    _assert_bit_exact(x16, d16, L=8, mu=16384, bits=15)


# ---------------------------------------------------------------------------
# Saturation: would-wrap cases must clip, not wrap
# ---------------------------------------------------------------------------

def test_sat16_scalar_would_wrap_cases():
    # 65534 wraps to -2 in int16; must saturate to 32767 instead.
    assert _sat16_scalar(65534) == (32767, True)
    assert _sat16_scalar(-65536) == (-32768, True)
    assert _sat16_scalar(32767) == (32767, False)
    assert _sat16_scalar(-32768) == (-32768, False)
    assert _sat16_scalar(0) == (0, False)


def test_sat16_vec_would_wrap_cases():
    v = np.array([40000, -40000, 123, 32768, -32769], np.int64)
    clipped, mask = _sat16_vec(v)
    assert np.array_equal(clipped, [32767, -32768, 123, 32767, -32768])
    assert np.array_equal(mask, [True, True, False, True, True])


def test_error_site_saturates_end_to_end():
    # Phase 1 trains the filter to output y = -x on full-scale input;
    # phase 2 flips d to +full-scale, so d - y ~= 65534 — far outside
    # int16. Wraparound would output ~-2; saturation must give 32767.
    rng = np.random.default_rng(13)
    n1, n2 = 3000, 50
    x1 = (rng.choice([-1, 1], n1) * 32000).astype(np.int16)
    d1 = (-x1).astype(np.int16)
    x2 = np.full(n2, 32000, np.int16)
    d2 = np.full(n2, 32000, np.int16)
    x16 = np.concatenate([x1, x2])
    d16 = np.concatenate([d1, d2])
    out = nlms_q15(x16, d16, L=4, mu_q15=16384, shadow_float=False)
    assert out["sat_counts"]["err"] > 0
    assert np.all(out["sat_positions"]["err"] >= n1)
    first = int(out["sat_positions"]["err"][0])
    assert out["e"][first] == 32767  # saturated, not wrapped


def test_gain_site_saturates_on_tiny_input_power():
    # Near-zero (but nonzero) window power with a large error: the
    # real-valued gain mu*e/(P+delta) >> 1 rails the Q15 gain.
    rng = np.random.default_rng(14)
    n = 500
    x16 = rng.choice([-4, 4], n).astype(np.int16)
    d16 = np.full(n, 20000, np.int16)
    out = nlms_q15(x16, d16, L=8, mu_q15=16384, shadow_float=False)
    assert out["sat_counts"]["gain"] > 0


def test_coeff_site_saturates_and_rails():
    # Echo gain -1.2 targets w[0] ~= -39322, below Q15 range: the
    # coefficient must pin at the -32768 rail with clip events counted.
    rng = np.random.default_rng(15)
    n = 4000
    x16 = (rng.choice([-1, 1], n) * 20000).astype(np.int16)
    d16 = np.clip(np.round(-1.2 * x16), -32768, 32767).astype(np.int16)
    out = nlms_q15(x16, d16, L=4, mu_q15=16384, shadow_float=False)
    assert out["sat_counts"]["coeff"] > 0
    assert out["n_sat_coeff_taps"] >= out["sat_counts"]["coeff"]
    assert out["w_final"][0] == -32768


# ---------------------------------------------------------------------------
# Convergence and stalling
# ---------------------------------------------------------------------------

def test_converges_to_known_path_at_q15_floor():
    x16, d16, h = _echo_signals(seed=3, n=20000)
    out = nlms_q15(x16, d16, L=32, mu_q15=16384, shadow_float=False)
    mis_db = _misalignment_db(out["w_final"] / 32768.0, h)
    # ||h||^2 ~= 0.45, L=32: rounding-only floor ~= -82 dB; truncation/
    # adaptation noise sits above it. Well below -25 dB is converged for
    # Q15; below -80 dB would mean we are somehow beating the noise
    # floor of the arithmetic, i.e. not actually running fixed point.
    assert -80.0 < mis_db < -25.0, f"misalignment {mis_db:.1f} dB"
    tail = slice(len(d16) // 2, None)
    e = out["e"].astype(np.float64)
    d = d16.astype(np.float64)
    assert np.mean(e[tail] ** 2) < 1e-2 * np.mean(d[tail] ** 2)


def test_stalls_after_convergence():
    # Full stalls (adaptation halts for a whole sample) require the gain
    # or every per-tap update to truncate to zero — a small-error, i.e.
    # post-initial-convergence, phenomenon: none should occur in the
    # first quarter of the run while the error is still large.
    x16, d16, _ = _echo_signals(seed=4, n=20000)
    out = nlms_q15(x16, d16, L=32, mu_q15=16384, shadow_float=False)
    assert out["n_stall_events"] > 0
    assert len(out["stall_positions"]) == out["n_stall_events"]
    assert np.median(out["stall_positions"]) > len(x16) // 4
    # Per-tap stalls (active tap, coefficient unchanged) are far more
    # numerous: they include every positive sub-LSB truncation at the
    # (g*x)>>15 stage, dominated by the small-|x| tap population.
    assert out["n_tap_stalls"] > out["n_stall_events"]
    assert out["n_tap_stalls"] == out["tap_stalls_per_sample"].sum()


def test_masking_degrades_without_full_stalls():
    # The stall detector evaluates the update *after* the mask, yet
    # coarser masking produces FEWER full-stall events, not more: floor
    # masking erases only positive sub-LSB updates, while negative ones
    # step a full effective LSB downward, keeping some tap moving and
    # the error (hence gain) elevated. The filter jitters in a limit
    # cycle instead of halting, so the degradation appears as a rising
    # misalignment floor — the word-length sweep's actual signal — not
    # as stall counts. Asserted here so the mechanism stays documented.
    x16, d16, h = _echo_signals(seed=5, n=20000)
    res = {}
    for bits in (15, 7):
        out = nlms_q15(x16, d16, L=32, mu_q15=16384, coeff_bits=bits,
                       shadow_float=False)
        res[bits] = (out["n_stall_events"],
                     _misalignment_db(out["w_final"] / 32768.0, h))
    assert res[7][0] < res[15][0], res  # the inversion, see comment
    # ~-44 dB at 15 bits vs ~+4 dB at 7 bits on this signal; a >= 20 dB
    # gap is far outside seed noise.
    assert res[7][1] > res[15][1] + 20.0, res


def test_silence_leaves_filter_untouched():
    x16 = np.zeros(1000, np.int16)
    d16 = np.full(1000, 3277, np.int16)  # ~0.1 full scale
    out = nlms_q15(x16, d16, L=32, mu_q15=16384, shadow_float=False)
    assert np.array_equal(out["e"], d16)  # nothing to cancel from
    assert np.all(out["w_final"] == 0)
    # All-zero window is skipped: no stall or saturation events.
    assert out["n_stall_events"] == 0
    assert all(v == 0 for v in out["sat_counts"].values())


# ---------------------------------------------------------------------------
# Word-length mask
# ---------------------------------------------------------------------------

def test_quantise_coeffs_identity_at_15_bits():
    w = np.array([32767, -32768, 1, -1, 12345], np.int64)
    assert np.array_equal(quantise_coeffs(w, 15), w)


def test_quantise_coeffs_masks_low_bits():
    w = np.array([32767, -32768, 255, -255, 256, -1], np.int64)
    got = quantise_coeffs(w, 7)  # effective LSB 2^8 = 256
    # Floor toward -inf: 255 -> 0 but -255 -> -256, -1 -> -256.
    assert np.array_equal(got, [32512, -32768, 0, -256, 256, -256])
    assert np.all(got % 256 == 0)


# ---------------------------------------------------------------------------
# Shadow float and trajectory
# ---------------------------------------------------------------------------

def test_shadow_float_matches_nlms_f64():
    # The in-loop shadow filter must be the same recursion as
    # aec_nlms.nlms on the identical quantised input. Tolerance, not
    # bit-exactness: nlms() estimates window power from a cumulative
    # sum, the shadow recomputes it exactly (both are ~1e-15 apart).
    x16, d16, _ = _echo_signals(seed=6, n=4000)
    out = nlms_q15(x16, d16, L=16, mu_q15=16384, shadow_float=True)
    ref = nlms(x16 / 32768.0, d16 / 32768.0, L=16, mu=0.5,
               delta=DELTA_Q30_DEFAULT / 2**30)
    np.testing.assert_allclose(out["w_float_final"], ref["w_final"],
                               atol=1e-10)
    # The recorded curve's final point equals the independently computed
    # divergence between the returned states.
    diff = out["w_final"] / 32768.0 - out["w_float_final"]
    expect = 10 * np.log10(np.sum(diff**2)
                           / np.sum(out["w_float_final"] ** 2))
    assert abs(float(out["coeff_div_db"][-1]) - expect) < 1e-4
    # After convergence the Q15 filter tracks the float one closely but
    # not perfectly: divergence should be a real, finite, negative dB.
    assert -100.0 < float(out["coeff_div_db"][-1]) < -10.0


def test_shadow_off_returns_none():
    x16, d16, _ = _echo_signals(seed=7, n=500)
    out = nlms_q15(x16, d16, L=8, mu_q15=16384, shadow_float=False)
    assert out["coeff_div_db"] is None
    assert out["w_float_final"] is None


def test_trajectory_shape_and_consistency():
    x16, d16, _ = _echo_signals(seed=8, n=1000)
    out = nlms_q15(x16, d16, L=8, mu_q15=16384, record_every=100,
                   shadow_float=False)
    assert out["w_traj"].shape == (10, 8)
    assert out["w_traj"].dtype == np.int16
    assert np.array_equal(out["w_traj"][-1], out["w_final"])
    out_none = nlms_q15(x16, d16, L=8, mu_q15=16384, shadow_float=False)
    assert out_none["w_traj"] is None
    assert np.array_equal(out_none["e"], out["e"])  # recording is inert
