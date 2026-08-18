"""Q15 fixed-point sample-wise NLMS adaptive filter (spec §7.3).

Same system as aec_nlms.nlms, re-implemented in fixed point. Signals and
coefficients are int16/Q15; every product is Q15 x Q15 -> Q30, accumulated
in int64, and every narrowing back to 16 bits saturates. The reference for
correctness is the §7.3 arithmetic executed naively (pure-Python unbounded
integers, tests/test_aec_nlms_fixed.py), against which this implementation
is bit-exact — not the float path, whose role here is the shadow filter for
the coefficient-divergence instrumentation.

Number formats
--------------
    signals x, d          int16, Q15 (the run's shared float->int16 scale)
    coefficients w        int16 values, held in an int64 array (bounded by
                          the 'coeff' saturation site; int64 storage keeps
                          the per-tap products from needing a cast)
    mu, gain g            Q15 integers
    window power P        int64, Q30 — exact integer sliding-window sum of
                          x^2; unlike the float path's cumulative sum this
                          is *exact* (no cancellation), so no drift guard
                          is needed. Bit-exact equality with the naive
                          reference (which recomputes P from the tap buffer
                          every sample) proves it.
    delta                 int64, Q30; default 1074 = round(1e-6 * 2^30),
                          the Q30 image of the float path's delta

Saturation enforcement
----------------------
All narrowing goes through _sat16_scalar / _sat16_vec — grep those two
names to enumerate every narrowing site. Between sites everything is
int64, where the bounds below make wraparound impossible. numpy is never
allowed to narrow silently: no int16/int32 casts exist in the arithmetic
path.

Rounding convention (this choice is load-bearing — see below)
-------------------------------------------------------------
Every product narrowing (y, gain, per-tap update) uses **magnitude
truncation** — shift toward zero via _trunc_shift_* — matching the
spec's description of sub-LSB updates "truncating to zero": under
magnitude truncation an update smaller than 1 LSB vanishes on *both*
signs, which is the stalling phenomenon §7.3 asks to instrument. The
word-length mask alone keeps the spec snippet's floor (>>/<<)
semantics.

A plain arithmetic shift (floor, toward -inf) at the update narrowing
was implemented first and fails catastrophically on real speech at full
15-bit precision: floor biases every update by -0.5 LSB in the mean,
and during speech pauses (tiny |x|, hence vanishing error feedback,
while err != 0 keeps updates firing) the per-tap bias accumulates
essentially unopposed — on the baseline scenario the mean coefficient
drifted to about -22000 of the -32768 rail against true taps of a few
hundred LSB, producing *positive* misalignment and negative ERLE. The
stalling/degradation behaviour of any fixed-point NLMS is therefore a
function of the narrowing convention, not just the word length; a
floor-truncating implementation behaves qualitatively differently from
this one. Reported in the study, not hidden here.

Arithmetic path and dynamic range (per sample)
----------------------------------------------
    y_acc = sum_k w[k] x[k]      |.| <= L * 2^30 (L <= 2^20 asserted,
                                 so |.| < 2^50)
    y     = sat16(trunc(y_acc, 15))                     [site 'y']
    err   = sat16(d - y)         |d - y| < 2^17         [site 'err']

(trunc(v, k) = magnitude truncation: |v| >> k with the sign restored.)

Normalisation — block floating point with a per-sample reciprocal.
Derivation: with M, E, X the Q15 integers for mu, e(n), x(n-k) and
P64 = P + delta in Q30, the real-valued NLMS update

    dw = mu_r e_r x_r / (P_r + delta_r)

expressed in Q15 units is exactly  dw_q15 = M E X / P64  (all 2^15/2^30
scale factors cancel). A Q15 DSP has no 64-bit divider, so one reciprocal
is computed per sample:

    s   = bitlen(P64) - 15       exponent; negative when P64 < 2^14
    m   = P64 >> s (or << -s)    mantissa in [2^14, 2^15)
    r   = 2^29 // m              in (2^14, 2^15]; the 2^15 endpoint (m
                                 exactly 2^14) clamps to 32767 — a <= 1 LSB
                                 reciprocal rounding, deliberately not
                                 counted as a saturation event
    g   = sat16(trunc(M E r, 14 + s))                   [site 'gain']
          |M E r| <= 2^15 * 2^16 * 2^15 = 2^46;  14 + s >= 0 since
          P64 >= 2 (P >= 1 when any tap is nonzero, delta >= 1)

The gain in Q15 units is g = 2^15 M E / P64 (from the derivation below:
1/P64 ~= r / 2^(29+s), so 2^15 M E r / 2^(29+s) narrowed by 14 + s).

g is the Q15 image of mu e / (P + delta). Narrowing it to Q15 *before*
the tap loop is the classic 16-bit-DSP structure (gain in a register, one
Q15 x Q15 MAC per tap) and is itself informative: when P is small the
real-valued gain exceeds Q15 full scale and rails at |g| ~= 1.0 — those
clips are counted at site 'gain', not hidden. The all-zero window
(P == 0) is skipped entirely: no tap can change, and counting the
structural gain rail during exact far-end silence would swamp the
instrumentation with meaningless events.

    dw[k] = trunc(g x[k], 15)    |g x| <= 2^31
    w[k]  = sat16(w[k] + dw[k])                         [site 'coeff']
    w     = quantise_coeffs(w, coeff_bits)   (identity at 15 bits)

Word-length mask
----------------
quantise_coeffs floors each coefficient to a multiple of 2^(15-bits)
toward -inf (arithmetic >> then <<, the spec's snippet). Applied after
every update, it simulates *storing* w at reduced precision. Floor
masking is asymmetric: a nonzero positive update smaller than the
effective LSB is erased (a stall), while a nonzero negative one steps a
full effective LSB downward. That asymmetry is a property of the
masking scheme under test and shows up in the word-length sweep; it is
documented, not corrected.

Instrumentation
---------------
Stalling is recorded at two granularities, both evaluated on the final
coefficient state of the sample — after saturation *and* after the mask,
so the word-length sweep and the stall counts tell one story:

  * Full-stall events (the spec's "adaptation halts"): sample n has
    err != 0, a nonzero tap in the window, and *no* coefficient changed.
    Covers g truncating to zero, every per-tap (g x)>>15 truncating to
    zero, and the mask erasing every change. A sample where saturation
    clamps every changed tap back to its rail also lands here (counted
    as both coeff-saturation and stall); with real signals that
    coincidence is vanishingly rare.
  * Per-tap stalls: a tap with x[k] != 0 during an attempted update
    (err != 0, g != 0 path taken, or g truncated to zero) whose
    coefficient did not change. Returned as a per-sample count array
    plus a total.

At full 15 bits the two agree in character: magnitude truncation kills
sub-LSB updates on both signs, so a converged filter stalls massively —
the spec's "adaptation halts" phenomenon at the Q15 noise floor. Under
the mask the two *diverge*, because the mask floors instead of
truncating toward zero: an update that survives the arithmetic path
(|dw| >= 1 Q15 LSB) but is smaller than the effective LSB is erased
when positive yet amplified to a full -LSB step when negative. Some tap
therefore keeps moving, the error (hence the gain) stays elevated, and
the filter jitters in a limit cycle around the solution instead of
halting. On a small stationary configuration (32 large taps, white
noise — see test_masking_degrades_without_full_stalls) that limit cycle
is bounded: fewer stall events than unmasked, graded misalignment
degradation. At full scale (3200 small-magnitude taps, real speech,
the benchmark's word-length sweep) the same asymmetry instead acts as
a one-way ratchet that walks the coefficients to the negative rail —
the identical floor-bias mechanism described above for update
arithmetic, re-entering through the coefficient store. Either way the
naive "coarser LSB => more stalling" expectation fails, and the
degradation the sweep measures lives in the accuracy metrics, not the
stall counters. A genuine property of the floor-masking scheme
(faithful to truncating two's-complement storage), reported as a
finding, not adjusted away.

Saturation: per-site event positions ('y', 'err', 'gain', 'coeff') and
counts; for 'coeff' one event per sample with any clipped tap, plus a
separate total clipped-tap count.

Shadow float: optionally runs the float64 NLMS recursion *inside the
same loop* on the identical input (x/2^15, d/2^15 — the same quantised
signals the Q15 path sees, so the divergence curve isolates arithmetic,
not input quantisation), and records the per-sample coefficient
divergence 10 log10(||w_q/2^15 - w_f||^2 / ||w_f||^2) in dB at O(1)
memory per sample. The shadow uses exact per-window power recomputation,
so it matches aec_nlms.nlms to float roundoff (~1e-15), not bit-exactly
(that path estimates power from a cumulative sum). The shadow is never
masked: at coeff_bits < 15 the curve measures total degradation against
the ideal float filter.
"""

from __future__ import annotations

import numpy as np

Q15_MAX = 32767
Q15_MIN = -32768
Q15_ONE = 32768  # 2^15
DELTA_Q30_DEFAULT = 1074  # round(1e-6 * 2^30)
_RECIP_SHIFT = 29
_L_MAX = 1 << 20  # keeps |y_acc| <= L * 2^30 < 2^50, far inside int64


def _trunc_shift_scalar(v: int, k: int) -> int:
    """Magnitude truncation: shift |v| right by k, restore the sign."""
    return -((-v) >> k) if v < 0 else v >> k


def _trunc_shift_vec(v: np.ndarray, k: int) -> np.ndarray:
    return np.where(v >= 0, v >> k, -((-v) >> k))


def _sat16_scalar(v: int) -> tuple[int, bool]:
    """Saturating narrowing of a Python int to int16 range."""
    if v > Q15_MAX:
        return Q15_MAX, True
    if v < Q15_MIN:
        return Q15_MIN, True
    return v, False


def _sat16_vec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saturating narrowing of an int64 vector; returns (clipped, mask)."""
    clipped = np.clip(v, Q15_MIN, Q15_MAX)
    return clipped, clipped != v


def quantise_coeffs(w_q15: np.ndarray, bits: int) -> np.ndarray:
    """Mask low coefficient bits (spec §7.3 snippet). bits=15 is identity.

    Arithmetic >>/<< floors each coefficient to a multiple of 2^(15-bits)
    toward -inf; see the module docstring for the resulting asymmetry.
    """
    if not 1 <= bits <= 15:
        raise ValueError(f"bits must be in [1, 15], got {bits}")
    shift = 15 - bits
    if shift == 0:
        return w_q15
    return (w_q15 >> shift) << shift


def nlms_q15(x16: np.ndarray, d16: np.ndarray, L: int, mu_q15: int,
             delta_q30: int = DELTA_Q30_DEFAULT, coeff_bits: int = 15,
             record_every: int | None = None,
             shadow_float: bool = True) -> dict:
    """Run Q15 NLMS over full int16 signals.

    Returns a dict with:
        e                int16 error/output signal
        w_final          int16 coefficients, newest-first (w_final[0]
                         pairs with x(n), matching the h_echo convention)
        w_traj           int16 (n_snapshots, L) or None; snapshot after
                         every record_every samples, newest-first
        n_stall_events, stall_positions
                         full-stall events: no coefficient changed
        n_tap_stalls     total per-tap stalls (active tap, no change)
        tap_stalls_per_sample
                         int32 per-sample per-tap stall counts
        sat_counts       {'y','err','gain','coeff'}: event counts
        sat_positions    same keys: sample indices (int64 arrays)
        n_sat_coeff_taps total individual clipped taps at site 'coeff'
        coeff_div_db     float32 per-sample divergence from the shadow
                         float filter (NaN before the shadow filter has
                         nonzero norm), or None if shadow_float=False
        w_float_final    shadow filter's float64 coefficients,
                         newest-first, or None
    """
    if x16.dtype != np.int16 or d16.dtype != np.int16:
        raise TypeError(
            f"x16/d16 must be int16, got {x16.dtype}/{d16.dtype}")
    if len(x16) != len(d16):
        raise ValueError(
            f"length mismatch: len(x)={len(x16)}, len(d)={len(d16)}")
    if not 0 < L <= _L_MAX:
        raise ValueError(f"L must be in (0, {_L_MAX}], got {L}")
    if not 1 <= mu_q15 <= Q15_MAX:
        raise ValueError(f"mu_q15 must be in [1, {Q15_MAX}], got {mu_q15}")
    if delta_q30 < 1:
        raise ValueError("delta_q30 must be >= 1 (guards the reciprocal)")
    if not 1 <= coeff_bits <= 15:
        raise ValueError(f"coeff_bits must be in [1, 15], got {coeff_bits}")
    n = len(x16)
    mask_shift = 15 - coeff_bits

    # Padded so xp[i : i+L] == [x(i-L+1), ..., x(i)] (oldest-first), same
    # convention as aec_nlms; coefficients are kept oldest-first too.
    x64 = x16.astype(np.int64)
    xp = np.concatenate([np.zeros(L - 1, np.int64), x64])
    xsq = x64 * x64  # each <= 2^30

    w = np.zeros(L, np.int64)  # values always in int16 range (site 'coeff')
    e = np.empty(n, np.int16)
    P = 0  # exact Q30 window power, Python int
    stall_pos: list[int] = []
    tap_stalls = np.zeros(n, np.int32)
    sat_pos: dict[str, list[int]] = {"y": [], "err": [], "gain": [],
                                     "coeff": []}
    n_sat_coeff_taps = 0
    snapshots: list[np.ndarray] = []

    if shadow_float:
        inv = 1.0 / Q15_ONE
        xpf = xp * inv
        df = d16 * inv
        mu_f = mu_q15 * inv
        delta_f = delta_q30 / (1 << 30)
        wf = np.zeros(L)
        div_db = np.full(n, np.nan, np.float32)

    for i in range(n):
        seg = xp[i:i + L]
        P += int(xsq[i])
        if i >= L:
            P -= int(xsq[i - L])

        y, sat = _sat16_scalar(_trunc_shift_scalar(int(seg @ w), 15))
        if sat:
            sat_pos["y"].append(i)
        err, sat = _sat16_scalar(int(d16[i]) - y)
        if sat:
            sat_pos["err"].append(i)
        e[i] = err

        # P > 0 iff any tap is nonzero (P is exactly the window's sum of
        # squares) — the all-zero window is skipped, see module docstring.
        if err != 0 and P > 0:
            p64 = P + delta_q30
            s = p64.bit_length() - 15
            m = p64 >> s if s >= 0 else p64 << -s
            r = (1 << _RECIP_SHIFT) // m
            if r > Q15_MAX:
                r = Q15_MAX  # m == 2^14 endpoint; reciprocal rounding
            g, sat = _sat16_scalar(_trunc_shift_scalar(
                mu_q15 * err * r, _RECIP_SHIFT + s - 15))
            if sat:
                sat_pos["gain"].append(i)
            if g != 0:
                dw = _trunc_shift_vec(g * seg, 15)
                w_cand, clip_mask = _sat16_vec(w + dw)
                n_clipped = int(np.count_nonzero(clip_mask))
                if n_clipped:
                    sat_pos["coeff"].append(i)
                    n_sat_coeff_taps += n_clipped
                if mask_shift:
                    w_cand = (w_cand >> mask_shift) << mask_shift
                stalled = (seg != 0) & (w_cand == w)
                tap_stalls[i] = np.count_nonzero(stalled)
                if np.array_equal(w_cand, w):
                    stall_pos.append(i)  # update erased post-sat, post-mask
                w = w_cand
            else:
                # Gain truncated to zero: every active tap stalls.
                tap_stalls[i] = np.count_nonzero(seg)
                stall_pos.append(i)

        if shadow_float:
            segf = xpf[i:i + L]
            err_f = df[i] - wf @ segf
            norm_f = float(segf @ segf)
            wf += (mu_f * err_f / (norm_f + delta_f)) * segf
            den = float(wf @ wf)
            if den > 0.0:
                diff = w * inv - wf
                num = float(diff @ diff)
                div_db[i] = 10.0 * np.log10(num / den) if num > 0.0 \
                    else -np.inf

        if record_every is not None and (i + 1) % record_every == 0:
            snapshots.append(w[::-1].astype(np.int16))

    return {
        "e": e,
        "w_final": w[::-1].astype(np.int16),
        "w_traj": (np.array(snapshots, np.int16)
                   if record_every is not None else None),
        "n_stall_events": len(stall_pos),
        "stall_positions": np.array(stall_pos, np.int64),
        "n_tap_stalls": int(tap_stalls.sum()),
        "tap_stalls_per_sample": tap_stalls,
        "sat_counts": {k: len(v) for k, v in sat_pos.items()},
        "sat_positions": {k: np.array(v, np.int64)
                          for k, v in sat_pos.items()},
        "n_sat_coeff_taps": n_sat_coeff_taps,
        "coeff_div_db": div_db if shadow_float else None,
        "w_float_final": wf[::-1].copy() if shadow_float else None,
    }
