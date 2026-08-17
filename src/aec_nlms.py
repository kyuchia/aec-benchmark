"""Float64 sample-wise NLMS adaptive filter.

    e(n)   = d(n) - w'(n) x(n)
    w(n+1) = w(n) + mu * e(n) * x(n) / (||x(n)||^2 + delta)

where x(n) is the vector of the last L reference samples. Deliberately
minimal: no double-talk detection (its absence is part of the study — it
makes SpeexDSP's built-in protection visible in the double-talk results),
no step-size scheduling, float64 throughout.

The sliding-window input power ||x(n)||^2 is evaluated in O(1) per sample
from a precomputed cumulative sum of x^2 — mathematically identical to
the recursive add/subtract update, without its accumulated drift.
"""

from __future__ import annotations

import numpy as np


def nlms(x: np.ndarray, d: np.ndarray, L: int, mu: float,
         delta: float = 1e-6,
         record_every: int | None = None) -> dict:
    """Run NLMS over full signals.

    Returns {'e': ndarray, 'w_final': ndarray, 'w_traj': ndarray | None}.
    w_traj (if recorded) has one row per record_every samples, each a full
    coefficient snapshot, oldest first; w_traj[k] is the state after
    sample (k+1)*record_every.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64)
    if len(x) != len(d):
        raise ValueError(f"length mismatch: len(x)={len(x)}, len(d)={len(d)}")
    if L <= 0:
        raise ValueError("L must be positive")
    n = len(x)

    # x padded so that xp[i : i+L] == [x(i-L+1), ..., x(i)]; coefficients are
    # kept in the same (oldest-first) order so the filter output is a plain
    # dot product with the slice.
    xp = np.concatenate([np.zeros(L - 1), x])
    # css[k] = sum of x[0..k-1]^2; window power in O(1) per sample.
    css = np.concatenate([[0.0], np.cumsum(x * x)])

    w = np.zeros(L)
    e = np.empty(n)
    snapshots: list[np.ndarray] = []

    for i in range(n):
        seg = xp[i:i + L]
        err = d[i] - w @ seg
        e[i] = err
        norm = css[i + 1] - css[max(0, i + 1 - L)]
        w += (mu * err / (norm + delta)) * seg
        if record_every is not None and (i + 1) % record_every == 0:
            snapshots.append(w[::-1].copy())

    # Internal storage is oldest-first; return newest-first (w[0] pairs with
    # x(n), matching the h_echo tap convention).
    return {
        "e": e,
        "w_final": w[::-1].copy(),
        "w_traj": np.array(snapshots) if record_every is not None else None,
    }
