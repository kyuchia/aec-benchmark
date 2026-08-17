"""Unit tests for the three-state segmentation.

A constructed signal with known activity block boundaries must yield
exactly the expected frame-state sequence, hangover included.
"""

import numpy as np

from segment import Segmentation, segment

FS = 16000
FRAME_MS = 20.0
FRAME = int(FRAME_MS * 1e-3 * FS)          # 320 samples
SEG_CFG = {
    "frame_ms": FRAME_MS,
    "energy_threshold_dbov": -45.0,
    "hangover_ms": 200.0,                   # 10 frames
}
HANG = 10
ACTIVE_LEVEL = 10.0 ** (-26.0 / 20.0)       # well above the -45 dBov gate


def _block_signal(blocks: list[tuple[float, float]], total_s: float) -> np.ndarray:
    """Constant-envelope noise in the given (start, stop) second intervals."""
    rng = np.random.default_rng(0)
    sig = np.zeros(int(total_s * FS))
    for t0, t1 in blocks:
        n = int((t1 - t0) * FS)
        sig[int(t0 * FS):int(t0 * FS) + n] = ACTIVE_LEVEL * np.sign(
            rng.standard_normal(n))
    return sig


def _frames(t0: float, t1: float) -> np.ndarray:
    """Frame indices covering [t0, t1) seconds."""
    return np.arange(int(t0 * 1000 / FRAME_MS), int(t1 * 1000 / FRAME_MS))


def test_known_block_layout_yields_exact_state_sequence():
    # 0-1 s silence | 1-2 s far only | 2-3 s overlap | 3-4 s near only |
    # 4-5 s silence.  (5 s => 250 frames)
    d_echo = _block_signal([(1.0, 3.0)], 5.0)
    s = _block_signal([(2.0, 4.0)], 5.0)
    seg = segment(d_echo, s, FS, SEG_CFG)

    assert seg.n_frames == 250
    expected_far = np.zeros(250, dtype=bool)
    expected_far[50:150] = True             # raw active
    expected_far[150:150 + HANG] = True     # hangover
    expected_near = np.zeros(250, dtype=bool)
    expected_near[100:200] = True
    expected_near[200:200 + HANG] = True

    np.testing.assert_array_equal(seg.far_active, expected_far)
    np.testing.assert_array_equal(seg.near_active, expected_near)

    # ERLE-valid = far AND NOT near: exactly the 1-2 s far-only second.
    expected_valid = np.zeros(250, dtype=bool)
    expected_valid[50:100] = True
    np.testing.assert_array_equal(seg.erle_valid, expected_valid)

    # Double-talk = the overlap second plus far's hangover into near's block.
    expected_dt = np.zeros(250, dtype=bool)
    expected_dt[100:150 + HANG] = True
    np.testing.assert_array_equal(seg.double_talk, expected_dt)


def test_hangover_bridges_short_pause_but_not_long_one():
    # Active 0-1 s, 100 ms pause (5 frames < hangover), active 1.1-2 s,
    # then a 400 ms pause (20 frames > hangover), active 2.4-3 s.
    sig = _block_signal([(0.0, 1.0), (1.1, 2.0), (2.4, 3.0)], 3.0)
    seg = segment(sig, np.zeros_like(sig), FS, SEG_CFG)

    # Short pause fully bridged.
    assert np.all(seg.far_active[_frames(1.0, 1.1)])
    # Long pause: held for exactly the hangover, then released.
    assert np.all(seg.far_active[_frames(2.0, 2.2)])         # 10-frame hold
    assert not np.any(seg.far_active[_frames(2.2, 2.4)])     # released
    assert np.all(seg.far_active[_frames(2.4, 3.0)])


def test_below_threshold_signal_is_inactive():
    quiet = 10.0 ** (-60.0 / 20.0)  # 15 dB below the gate
    sig = quiet * np.ones(FS)
    seg = segment(sig, np.zeros_like(sig), FS, SEG_CFG)
    assert not np.any(seg.far_active)
    assert not np.any(seg.near_active)
    assert not np.any(seg.erle_valid)


def test_frame_times_centred():
    seg = Segmentation(frame_len=320, sample_rate=FS,
                       far_active=np.zeros(3, dtype=bool),
                       near_active=np.zeros(3, dtype=bool))
    np.testing.assert_allclose(seg.frame_times_s(), [0.01, 0.03, 0.05])
