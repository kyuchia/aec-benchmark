"""Three-state activity segmentation from ground-truth components.

States per analysis frame: far-active, near-active, or neither — derived
from the ground-truth component signals (d_echo, the echo at the mic, and
s, the reverberant near-end at the mic) independently. Detection never
runs on the mixture: the components are known exactly, so this is not a
VAD estimation problem.

Rule: a frame is active when its mean power exceeds an absolute energy
threshold (dBov, config); activity is then held for a hangover period
after the last raw-active frame, bridging inter-word pauses so that a
brief gap inside an utterance does not flip the state.

ERLE-valid frames are far-active AND NOT near-active: during double-talk
the error signal legitimately contains near-end speech, and during far
silence ERLE is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segmentation:
    frame_len: int              # samples per frame
    sample_rate: int
    far_active: np.ndarray      # (n_frames,) bool
    near_active: np.ndarray     # (n_frames,) bool

    @property
    def n_frames(self) -> int:
        return len(self.far_active)

    @property
    def erle_valid(self) -> np.ndarray:
        return self.far_active & ~self.near_active

    @property
    def double_talk(self) -> np.ndarray:
        return self.far_active & self.near_active

    def frame_times_s(self) -> np.ndarray:
        """Centre time of each frame."""
        return (np.arange(self.n_frames) + 0.5) * self.frame_len / self.sample_rate


def frame_power(sig: np.ndarray, frame_len: int) -> np.ndarray:
    """Mean power per non-overlapping frame (trailing partial frame dropped)."""
    n_frames = len(sig) // frame_len
    frames = sig[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.mean(frames**2, axis=1)


def _active_frames(sig: np.ndarray, frame_len: int, threshold_dbov: float,
                   hang_frames: int) -> np.ndarray:
    power = frame_power(sig, frame_len)
    raw = power > 10.0 ** (threshold_dbov / 10.0)
    if hang_frames <= 0:
        return raw
    active = raw.copy()
    last_raw = -(hang_frames + 1)
    for i, is_raw in enumerate(raw):
        if is_raw:
            last_raw = i
        elif i - last_raw <= hang_frames:
            active[i] = True
    return active


def segment(d_echo: np.ndarray, s: np.ndarray, sample_rate: int,
            seg_cfg: dict) -> Segmentation:
    frame_len = int(round(seg_cfg["frame_ms"] * 1e-3 * sample_rate))
    threshold_dbov = float(seg_cfg["energy_threshold_dbov"])
    hang_frames = int(round(seg_cfg["hangover_ms"] / seg_cfg["frame_ms"]))

    return Segmentation(
        frame_len=frame_len,
        sample_rate=sample_rate,
        far_active=_active_frames(d_echo, frame_len, threshold_dbov,
                                  hang_frames),
        near_active=_active_frames(s, frame_len, threshold_dbov, hang_frames),
    )
