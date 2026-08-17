"""Room simulation: geometry, RIR generation, calibrated RT60.

Rooms are pyroomacoustics ShoeBoxes driven by the image source method.
inverse_sabine is used only to initialise the wall absorption; the value
actually used is calibrated by bisection until the RT60 measured on the
generated loudspeaker->mic RIR (Schroeder backward integration) is within
a configured tolerance of the target, at a configured reference distance.
The same calibrated absorption is applied across the other distance
levels of that RT60 row; achieved RT60 is measured and stored for every
scenario regardless.

Calibration results are cached in data/generated/rt60_calibration.json,
keyed by target RT60 and fingerprinted against the room configuration so
a config change invalidates the cache.

The propagation delay is deliberately left in the RIRs — delay handling
is part of what an AEC is being evaluated on — and an assertion guards
against its accidental removal.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

SPEED_OF_SOUND_M_S = 343.0

# Tolerance for "at an exact half-dimension" / "axis-aligned" degeneracy
# checks, in metres.
_GEOMETRY_EPS_M = 0.02

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CALIBRATION_CACHE = _REPO_ROOT / "data" / "generated" / "rt60_calibration.json"

_MAX_BISECTION_STEPS = 30


@dataclass
class RoomRirs:
    h_echo: np.ndarray          # loudspeaker -> mic
    h_near: np.ndarray          # near-end talker -> mic
    rt60_target_s: float
    rt60_achieved_echo_s: float
    rt60_achieved_near_s: float
    calibration: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _position_at(mic: np.ndarray, azimuth_deg: float, distance_m: float,
                 height_m: float) -> np.ndarray:
    """Point at a 3D distance from the mic along an azimuth.

    The horizontal offset is shrunk so that the full 3D distance from the
    mic equals distance_m despite the height difference.
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


def _resolve_geometry(room_cfg: dict, speaker_mic_distance_m: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return dims, mic, speaker, talker


# ---------------------------------------------------------------------------
# RIR computation and RT60 measurement
# ---------------------------------------------------------------------------

def _compute_rirs(dims: np.ndarray, absorption: float, max_order: int,
                  sample_rate: int, mic: np.ndarray, speaker: np.ndarray,
                  talker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shoebox = pra.ShoeBox(
        dims,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    shoebox.add_source(speaker)
    shoebox.add_source(talker)
    shoebox.add_microphone(mic)
    shoebox.compute_rir()
    return (np.asarray(shoebox.rir[0][0], dtype=np.float64),
            np.asarray(shoebox.rir[0][1], dtype=np.float64))


def _measure_rt60(h: np.ndarray, fs: int) -> float:
    """Achieved RT60 from an RIR via Schroeder backward integration."""
    return float(pra.experimental.rt60.measure_rt60(h, fs=fs))


def _first_arrival_index(h: np.ndarray) -> int:
    threshold = 0.05 * np.max(np.abs(h))
    return int(np.argmax(np.abs(h) > threshold))


# ---------------------------------------------------------------------------
# Absorption calibration
# ---------------------------------------------------------------------------

def _room_fingerprint(room_cfg: dict, sample_rate: int) -> str:
    payload = json.dumps({"room": room_cfg, "fs": sample_rate}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if _CALIBRATION_CACHE.exists():
        with open(_CALIBRATION_CACHE) as f:
            return json.load(f)
    return {}


def _store_cache(cache: dict) -> None:
    _CALIBRATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CALIBRATION_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def calibrate_rt60(room_cfg: dict, rt60_target_s: float,
                   sample_rate: int) -> dict:
    """Bisect wall absorption until measured RT60 hits the target.

    Measured on h_echo at the configured reference distance. Asserts the
    final value is within the configured tolerance — a future regression
    (geometry change, library change) fails loudly at scenario-build time.
    """
    cal_cfg = room_cfg["rt60_calibration"]
    tolerance = float(cal_cfg["tolerance_pct"]) / 100.0
    reference_distance = float(cal_cfg["reference_distance_m"])
    fingerprint = _room_fingerprint(room_cfg, sample_rate)
    key = f"{rt60_target_s:g}"

    cache = _load_cache()
    entry = cache.get(key)
    if entry is not None and entry.get("fingerprint") == fingerprint:
        return entry

    dims, mic, speaker, talker = _resolve_geometry(room_cfg,
                                                   reference_distance)
    absorption_init, max_order = pra.inverse_sabine(rt60_target_s, dims)

    def measure(absorption: float) -> float:
        h_echo, _ = _compute_rirs(dims, absorption, max_order, sample_rate,
                                  mic, speaker, talker)
        return _measure_rt60(h_echo, sample_rate)

    def within(measured: float) -> bool:
        return abs(measured - rt60_target_s) <= tolerance * rt60_target_s

    rt60_init = measure(absorption_init)
    absorption, measured = absorption_init, rt60_init

    if not within(measured):
        # RT60 decreases monotonically as absorption increases. Bracket the
        # target: a_lo yields RT60 above target, a_hi below.
        if measured > rt60_target_s:
            a_lo = absorption_init
            a_hi = absorption_init
            for _ in range(_MAX_BISECTION_STEPS):
                a_hi = min(a_hi * 1.5, 0.9999)
                if measure(a_hi) < rt60_target_s or a_hi >= 0.9999:
                    break
        else:
            a_hi = absorption_init
            a_lo = absorption_init
            for _ in range(_MAX_BISECTION_STEPS):
                a_lo = a_lo / 1.5
                if measure(a_lo) > rt60_target_s:
                    break
        for _ in range(_MAX_BISECTION_STEPS):
            absorption = 0.5 * (a_lo + a_hi)
            measured = measure(absorption)
            if within(measured):
                break
            if measured > rt60_target_s:
                a_lo = absorption
            else:
                a_hi = absorption

    if not within(measured):
        raise AssertionError(
            f"RT60 calibration failed for target {rt60_target_s} s: best "
            f"achieved {measured:.3f} s with absorption {absorption:.4f} "
            f"(tolerance ±{tolerance * 100:.0f}%)"
        )

    entry = {
        "rt60_target_s": rt60_target_s,
        "absorption_calibrated": float(absorption),
        "absorption_sabine_init": float(absorption_init),
        "rt60_achieved_sabine_init_s": float(rt60_init),
        "rt60_achieved_calibrated_s": float(measured),
        "reference_distance_m": reference_distance,
        "max_order": int(max_order),
        "tolerance_pct": float(cal_cfg["tolerance_pct"]),
        "fingerprint": fingerprint,
    }
    cache[key] = entry
    _store_cache(cache)
    return entry


# ---------------------------------------------------------------------------
# Scenario RIRs
# ---------------------------------------------------------------------------

def build_rirs(room_cfg: dict, rt60_target_s: float,
               speaker_mic_distance_m: float, sample_rate: int) -> RoomRirs:
    dims, mic, speaker, talker = _resolve_geometry(room_cfg,
                                                   speaker_mic_distance_m)
    calibration = calibrate_rt60(room_cfg, rt60_target_s, sample_rate)

    h_echo, h_near = _compute_rirs(
        dims, calibration["absorption_calibrated"], calibration["max_order"],
        sample_rate, mic, speaker, talker)

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
        "near_talker_distance_m": room_cfg["near_talker"]["distance_m"],
        "absorption": calibration["absorption_calibrated"],
        "max_order": calibration["max_order"],
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
        calibration=calibration,
        geometry=geometry,
    )
