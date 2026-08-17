"""Signal synthesis: speech selection, levelling, mixing, int16 conversion.

Signal model (all at the config sample rate):

    x        far-end signal (what the loudspeaker plays)
    d_echo   x convolved with h_echo
    s_clean  near-end speech before its room path
    s        s_clean convolved with h_near
    v        background noise at the target SNR
    d        d_echo + s + v (microphone signal)

Levels: speech material is normalised to a fixed active-speech level
using an energy-gated RMS (a simplified active-level measure, not ITU-T
P.56): frames whose RMS falls more than a configured threshold below the
peak frame RMS are treated as inactive. The near-end level is then pinned
to the configured SER against the echo, measured over the double-talk
overlap, so the fixed-point operating point is controlled rather than an
accident of room geometry.

int16 conversion: one scaling constant per run, computed from max(|x|,|d|)
with configured headroom, applied identically to every system in the run,
and asserted not to clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

INT16_FULL_SCALE = 32767.0


# ---------------------------------------------------------------------------
# Speech material selection (reproducible, seeded)
# ---------------------------------------------------------------------------

def list_speakers(dataset_dir: Path) -> list[str]:
    speakers = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if not speakers:
        raise FileNotFoundError(f"no speaker directories under {dataset_dir}")
    return speakers


def select_speaker_pairs(dataset_dir: Path, n_seeds: int,
                         selection_seed: int) -> list[tuple[str, str]]:
    """Draw n_seeds disjoint (far, near) speaker pairs.

    All 2*n_seeds speakers are distinct, so no material is shared between
    seeds and far/near are never the same speaker.
    """
    speakers = list_speakers(dataset_dir)
    if len(speakers) < 2 * n_seeds:
        raise ValueError(f"need {2 * n_seeds} speakers, found {len(speakers)}")
    rng = np.random.default_rng(selection_seed)
    chosen = rng.choice(len(speakers), size=2 * n_seeds, replace=False)
    return [
        (speakers[chosen[2 * i]], speakers[chosen[2 * i + 1]])
        for i in range(n_seeds)
    ]


def load_speaker_material(dataset_dir: Path, speaker: str, need_s: float,
                          sample_rate: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Concatenate a speaker's utterances (seeded order) to cover need_s.

    Returns the trimmed signal and the list of utterance IDs actually used.
    """
    files = sorted((dataset_dir / speaker).rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"no flac files for speaker {speaker}")
    order = rng.permutation(len(files))

    need_n = int(round(need_s * sample_rate))
    chunks: list[np.ndarray] = []
    used: list[str] = []
    total = 0
    for idx in order:
        audio, sr = sf.read(files[idx], dtype="float64")
        if audio.ndim != 1:
            audio = audio[:, 0]
        if sr != sample_rate:
            audio = sps.resample_poly(audio, sample_rate, sr)
        chunks.append(audio)
        used.append(files[idx].stem)
        total += len(audio)
        if total >= need_n:
            break
    if total < need_n:
        raise ValueError(
            f"speaker {speaker}: only {total / sample_rate:.1f} s of material, "
            f"need {need_s:.1f} s"
        )
    return np.concatenate(chunks)[:need_n], used


# ---------------------------------------------------------------------------
# Active-level normalisation (simplified, energy-gated RMS — not P.56)
# ---------------------------------------------------------------------------

def active_rms(sig: np.ndarray, sample_rate: int, frame_ms: float,
               threshold_db: float) -> float:
    """RMS over frames within threshold_db of the peak frame RMS."""
    frame_len = int(round(frame_ms * 1e-3 * sample_rate))
    n_frames = len(sig) // frame_len
    if n_frames == 0:
        raise ValueError("signal shorter than one frame")
    frames = sig[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_power = np.mean(frames**2, axis=1)
    peak_power = np.max(frame_power)
    if peak_power <= 0.0:
        raise ValueError("signal is silent; cannot measure active level")
    active = frame_power > peak_power * 10.0 ** (-threshold_db / 10.0)
    return float(np.sqrt(np.mean(frame_power[active])))


def normalise_active_level(sig: np.ndarray, sample_rate: int,
                           levels_cfg: dict) -> np.ndarray:
    target_rms = 10.0 ** (levels_cfg["active_level_dbov"] / 20.0)
    rms = active_rms(
        sig,
        sample_rate,
        levels_cfg["active_gate_frame_ms"],
        levels_cfg["active_gate_threshold_db"],
    )
    return sig * (target_rms / rms)


# ---------------------------------------------------------------------------
# Timeline placement
# ---------------------------------------------------------------------------

def _segments_to_samples(segments: list[list[float]],
                         sample_rate: int) -> list[tuple[int, int]]:
    return [
        (int(round(t0 * sample_rate)), int(round(t1 * sample_rate)))
        for t0, t1 in segments
    ]


def total_active_seconds(segments: list[list[float]]) -> float:
    return sum(t1 - t0 for t0, t1 in segments)


def place_material(material: np.ndarray, segments: list[list[float]],
                   n_samples: int, sample_rate: int) -> np.ndarray:
    """Lay consecutive material into the active segments of a zero signal."""
    out = np.zeros(n_samples)
    cursor = 0
    for i0, i1 in _segments_to_samples(segments, sample_rate):
        length = i1 - i0
        out[i0:i1] = material[cursor:cursor + length]
        cursor += length
    return out


def activity_mask(segments: list[list[float]], n_samples: int,
                  sample_rate: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for i0, i1 in _segments_to_samples(segments, sample_rate):
        mask[i0:i1] = True
    return mask


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def make_noise(noise_type: str, n_samples: int, reference_speech: np.ndarray,
               noise_cfg: dict, sample_rate: int,
               rng: np.random.Generator) -> np.ndarray:
    """Unit-variance-ish noise of the requested colour (scaled later)."""
    white = rng.standard_normal(n_samples)
    if noise_type == "white":
        return white
    if noise_type == "speech_shaped":
        nperseg = int(noise_cfg["spectrum_nperseg"])
        freqs, psd = sps.welch(reference_speech, fs=sample_rate,
                               nperseg=nperseg)
        gains = np.sqrt(psd)
        gains /= np.max(gains)
        fir = sps.firwin2(int(noise_cfg["filter_taps"]),
                          freqs / (sample_rate / 2.0), gains)
        return sps.fftconvolve(white, fir, mode="same")
    raise ValueError(f"unknown noise_type: {noise_type}")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@dataclass
class SignalSet:
    x: np.ndarray
    d_echo: np.ndarray
    s_clean: np.ndarray
    s: np.ndarray
    v: np.ndarray
    d: np.ndarray
    far_mask: np.ndarray        # far-end source active (timeline, not VAD)
    near_mask: np.ndarray
    meta: dict = field(default_factory=dict)


def synthesise(cfg: dict, scenario: dict, seed_index: int,
               h_echo: np.ndarray, h_near: np.ndarray,
               noise_seed: int | None = None) -> SignalSet:
    """Build the full signal set for one (scenario, seed) cell.

    noise_seed overrides the default derived noise stream — the batch
    driver passes a value derived from the cell identity so the noise
    realisation is deterministic per cell and, critically, identical for
    every system evaluated on that cell.
    """
    sample_rate = int(cfg["sample_rate"])
    duration_s = float(cfg["duration_s"])
    n = int(round(duration_s * sample_rate))
    levels_cfg = cfg["levels"]
    speech_cfg = cfg["speech"]
    dataset_dir = Path(speech_cfg["dataset_dir"])

    timeline = cfg["timelines"][scenario["talk"]]
    far_segments = timeline["far"]
    near_segments = timeline["near"]
    for t0, t1 in far_segments + near_segments:
        if not (0.0 <= t0 < t1 <= duration_s):
            raise ValueError(f"segment [{t0}, {t1}] outside [0, {duration_s}]")

    pairs = select_speaker_pairs(dataset_dir, int(speech_cfg["n_seeds"]),
                                 int(speech_cfg["speaker_selection_seed"]))
    far_speaker, near_speaker = pairs[seed_index]

    meta: dict = {
        "seed_index": seed_index,
        "far_speaker": far_speaker,
        "near_speaker": near_speaker,
        "far_files": [],
        "near_files": [],
    }

    # Independent, deterministic streams for utterance order and noise.
    base = [int(speech_cfg["speaker_selection_seed"]), seed_index]
    far_rng = np.random.default_rng(base + [1])
    near_rng = np.random.default_rng(base + [2])
    noise_rng = np.random.default_rng(
        base + [3] if noise_seed is None else [noise_seed])

    reference_speech = []  # for speech-shaped noise spectrum

    x = np.zeros(n)
    if far_segments:
        material, files = load_speaker_material(
            dataset_dir, far_speaker, total_active_seconds(far_segments),
            sample_rate, far_rng)
        material = normalise_active_level(material, sample_rate, levels_cfg)
        x = place_material(material, far_segments, n, sample_rate)
        meta["far_files"] = files
        reference_speech.append(material)

    s_clean = np.zeros(n)
    if near_segments:
        material, files = load_speaker_material(
            dataset_dir, near_speaker, total_active_seconds(near_segments),
            sample_rate, near_rng)
        material = normalise_active_level(material, sample_rate, levels_cfg)
        s_clean = place_material(material, near_segments, n, sample_rate)
        meta["near_files"] = files
        reference_speech.append(material)

    d_echo = sps.fftconvolve(x, h_echo)[:n]
    s = sps.fftconvolve(s_clean, h_near)[:n]

    far_mask = activity_mask(far_segments, n, sample_rate)
    near_mask = activity_mask(near_segments, n, sample_rate)

    # Pin the near-end level to the configured SER (near over echo) on the
    # double-talk overlap. Only meaningful when both sources are present.
    overlap = far_mask & near_mask
    ser_db = scenario["ser_db"]
    if np.any(overlap) and np.any(s[overlap] != 0.0):
        p_echo = float(np.mean(d_echo[overlap] ** 2))
        p_near = float(np.mean(s[overlap] ** 2))
        gain = np.sqrt(p_echo * 10.0 ** (ser_db / 10.0) / p_near)
        s = s * gain
        s_clean = s_clean * gain
        meta["ser_db_configured"] = float(ser_db)
        meta["ser_gain_applied"] = float(gain)
        meta["ser_db_achieved"] = float(
            10.0 * np.log10(np.mean(s[overlap] ** 2) / p_echo))
    else:
        meta["ser_db_configured"] = None

    v = np.zeros(n)
    noise_type = scenario["noise_type"]
    meta["noise_type"] = noise_type
    if noise_type != "none":
        snr_db = scenario["snr_db"]
        if snr_db is None:
            raise ValueError("snr_db must be set when noise_type != none")
        speech_at_mic = d_echo + s
        speech_mask = far_mask | near_mask
        if not np.any(speech_mask):
            raise ValueError("cannot set SNR: no speech activity in timeline")
        v = make_noise(noise_type, n, np.concatenate(reference_speech),
                       cfg["noise"], sample_rate, noise_rng)
        p_speech = float(np.mean(speech_at_mic[speech_mask] ** 2))
        p_noise = float(np.mean(v**2))
        v = v * np.sqrt(p_speech * 10.0 ** (-snr_db / 10.0) / p_noise)
        meta["snr_db_configured"] = float(snr_db)

    d = d_echo + s + v

    meta["active_level_dbov"] = levels_cfg["active_level_dbov"]
    return SignalSet(x=x, d_echo=d_echo, s_clean=s_clean, s=s, v=v, d=d,
                     far_mask=far_mask, near_mask=near_mask, meta=meta)


# ---------------------------------------------------------------------------
# int16 conversion — one scale per run, clipping asserted
# ---------------------------------------------------------------------------

def compute_int16_scale(signals: list[np.ndarray], headroom_db: float) -> float:
    """Scale putting the loudest input signal headroom_db below full scale."""
    peak = max(float(np.max(np.abs(s))) for s in signals)
    if peak <= 0.0:
        raise ValueError("all signals are silent; cannot compute scale")
    return INT16_FULL_SCALE * 10.0 ** (-headroom_db / 20.0) / peak


def float_to_int16(sig: np.ndarray, scale: float, name: str = "signal") -> np.ndarray:
    scaled = sig * scale
    peak = float(np.max(np.abs(scaled))) if len(scaled) else 0.0
    if peak > INT16_FULL_SCALE:
        raise AssertionError(
            f"int16 conversion would clip {name}: peak {peak:.1f} > "
            f"{INT16_FULL_SCALE}"
        )
    return np.round(scaled).astype(np.int16)


def int16_to_float(sig: np.ndarray, scale: float) -> np.ndarray:
    return sig.astype(np.float64) / scale
