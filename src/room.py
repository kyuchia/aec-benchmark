"""Room simulation: geometry, RIR generation, achieved-RT60 measurement.

Rooms are pyroomacoustics ShoeBoxes driven by the image source method,
with absorption and reflection order derived from the target RT60 via
inverse_sabine. The propagation delay is deliberately left in the RIRs —
delay handling is part of what an AEC is being evaluated on.

The achieved RT60 is measured from the generated RIR (Schroeder backward
integration via pyroomacoustics) and reported alongside the target;
inverse_sabine plus ISM routinely lands off the requested value, so
figures must be labelled with achieved numbers, not targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyroomacoustics as pra

SPEED_OF_SOUND_M_S = 343.0

# Tolerance for "at an exact half-dimension" / "axis-aligned" degeneracy
# checks, in metres.
_GEOMETRY_EPS_M = 0.02


@dataclass
class RoomRirs:
    h_echo: np.ndarray          # loudspeaker -> mic
    h_near: np.ndarray          # near-end talker -> mic
    rt60_target_s: float
    rt60_achieved_echo_s: float
    rt60_achieved_near_s: float
    geometry: dict = field(default_factory=dict)


def _position_at(mic: np.ndarray, azimuth_deg: float, distance_m: float,
                 height_m: float) -> np.ndarray:
    """Point at a horizontal distance from the mic along an azimuth.

    The distance constraint is applied in 3D: the horizontal offset is
    shrunk so that the full 3D distance from the mic equals distance_m.
    """
    dz = height_m - mic[2]
    if abs(dz) >= distance_m:
        raise ValueError(
            f"height offset {dz} m exceeds requested distance {distance_m} m"
        )
    horizontal = math.sqrt(distance_m**2 - dz**2)
    az = math.radians(azimuth_deg)
    return np.array(
        [mic[0] + horizontal * math.cos(az),
         mic[1] + horizontal * math.sin(az),
         height_m]
    )


def _assert_point_valid(name: str, pos: np.ndarray, dims: np.ndarray,
                        clearance_m: float) -> None:
    for axis, (coord, dim) in enumerate(zip(pos, dims)):
        if coord < clearance_m or coord > dim - clearance_m:
            raise ValueError(
                f"{name} axis {axis}: coordinate {coord:.3f} m violates the "
                f"{clearance_m} m wall clearance (room dim {dim} m)"
            )
        if abs(coord - dim / 2.0) < _GEOMETRY_EPS_M:
            raise ValueError(
                f"{name} axis {axis}: coordinate {coord:.3f} m sits at the "
                f"room half-dimension {dim / 2.0} m (degenerate symmetry)"
            )


def _assert_not_axis_aligned(name: str, pos: np.ndarray,
                             mic: np.ndarray) -> None:
    shared = sum(1 for a, b in zip(pos, mic) if abs(a - b) < _GEOMETRY_EPS_M)
    if shared >= 2:
        raise ValueError(
            f"{name} at {pos} is axis-aligned with the mic at {mic} "
            f"({shared} shared coordinates; degenerate image-source symmetry)"
        )


def _measure_rt60(h: np.ndarray, fs: int) -> float:
    """Achieved RT60 from an RIR via Schroeder backward integration."""
    return float(pra.experimental.rt60.measure_rt60(h, fs=fs))


def _first_arrival_index(h: np.ndarray) -> int:
    threshold = 0.05 * np.max(np.abs(h))
    return int(np.argmax(np.abs(h) > threshold))


def build_rirs(room_cfg: dict, rt60_target_s: float,
               speaker_mic_distance_m: float, sample_rate: int) -> RoomRirs:
    dims = np.array(room_cfg["dimensions_m"], dtype=float)
    mic = np.array(room_cfg["mic_position_m"], dtype=float)
    clearance = float(room_cfg["min_wall_clearance_m"])

    ls_cfg = room_cfg["loudspeaker"]
    nt_cfg = room_cfg["near_talker"]
    speaker = _position_at(mic, ls_cfg["azimuth_deg"], speaker_mic_distance_m,
                           ls_cfg["height_m"])
    talker = _position_at(mic, nt_cfg["azimuth_deg"], nt_cfg["distance_m"],
                          nt_cfg["height_m"])

    _assert_point_valid("mic", mic, dims, clearance)
    _assert_point_valid("loudspeaker", speaker, dims, clearance)
    _assert_point_valid("near_talker", talker, dims, clearance)
    _assert_not_axis_aligned("loudspeaker", speaker, mic)
    _assert_not_axis_aligned("near_talker", talker, mic)

    for name, pos, want in [
        ("loudspeaker", speaker, speaker_mic_distance_m),
        ("near_talker", talker, nt_cfg["distance_m"]),
    ]:
        got = float(np.linalg.norm(pos - mic))
        if abs(got - want) > 1e-9:
            raise AssertionError(f"{name} distance {got} != requested {want}")

    e_absorption, max_order = pra.inverse_sabine(rt60_target_s, dims)
    shoebox = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    shoebox.add_source(speaker)
    shoebox.add_source(talker)
    shoebox.add_microphone(mic)
    shoebox.compute_rir()

    h_echo = np.asarray(shoebox.rir[0][0], dtype=np.float64)
    h_near = np.asarray(shoebox.rir[0][1], dtype=np.float64)

    # Guard against any accidental delay removal: the direct path must not
    # arrive before pure propagation allows.
    expected_delay = speaker_mic_distance_m / SPEED_OF_SOUND_M_S * sample_rate
    first_arrival = _first_arrival_index(h_echo)
    if first_arrival <= 0 or first_arrival < 0.8 * expected_delay:
        raise AssertionError(
            f"h_echo first arrival at sample {first_arrival}, expected "
            f">= {expected_delay:.1f} for {speaker_mic_distance_m} m — "
            "propagation delay appears to have been removed"
        )

    geometry = {
        "room_dimensions_m": dims.tolist(),
        "mic_position_m": mic.tolist(),
        "loudspeaker_position_m": speaker.tolist(),
        "near_talker_position_m": talker.tolist(),
        "speaker_mic_distance_m": speaker_mic_distance_m,
        "near_talker_distance_m": nt_cfg["distance_m"],
        "e_absorption": float(e_absorption),
        "max_order": int(max_order),
        "expected_direct_delay_samples": float(expected_delay),
        "h_echo_first_arrival_sample": first_arrival,
        "h_echo_peak_sample": int(np.argmax(np.abs(h_echo))),
        "h_echo_peak_value": float(h_echo[np.argmax(np.abs(h_echo))]),
    }

    return RoomRirs(
        h_echo=h_echo,
        h_near=h_near,
        rt60_target_s=rt60_target_s,
        rt60_achieved_echo_s=_measure_rt60(h_echo, sample_rate),
        rt60_achieved_near_s=_measure_rt60(h_near, sample_rate),
        geometry=geometry,
    )
