"""Unit tests for signal synthesis: levelling, SER, SNR, int16 conversion."""

from pathlib import Path

import numpy as np
import pytest
import yaml

import signals as sg

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "speech" / "LibriSpeech" / "test-clean"

needs_dataset = pytest.mark.skipif(
    not DATASET_DIR.is_dir(),
    reason="LibriSpeech test-clean not fetched (run scripts/fetch_data.py)",
)


def _load_cfg() -> dict:
    with open(REPO_ROOT / "config" / "scenarios.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["speech"]["dataset_dir"] = str(DATASET_DIR)
    return cfg


def test_active_level_normalisation_hits_target():
    # Bursty synthetic signal: speech-like activity with long silences.
    rng = np.random.default_rng(1)
    fs = 16000
    sig = np.zeros(10 * fs)
    sig[fs : 3 * fs] = rng.standard_normal(2 * fs)
    sig[6 * fs : 7 * fs] = 0.3 * rng.standard_normal(fs)
    levels_cfg = {
        "active_level_dbov": -26.0,
        "active_gate_frame_ms": 20.0,
        "active_gate_threshold_db": 40.0,
    }
    out = sg.normalise_active_level(sig, fs, levels_cfg)
    rms = sg.active_rms(out, fs, 20.0, 40.0)
    assert abs(20 * np.log10(rms) - (-26.0)) < 0.1
    # Silence must stay silence: gating ignores it rather than biasing level.
    assert np.all(out[: fs - 160] == 0)


def test_float_to_int16_asserts_on_clipping():
    sig = np.array([0.5, -1.1])
    with pytest.raises(AssertionError, match="clip"):
        sg.float_to_int16(sig, scale=32767.0)


def test_int16_round_trip_is_lossy_but_scaled():
    rng = np.random.default_rng(2)
    sig = 0.1 * rng.standard_normal(1000)
    scale = sg.compute_int16_scale([sig], headroom_db=6.0)
    back = sg.int16_to_float(sg.float_to_int16(sig, scale), scale)
    assert np.max(np.abs(back - sig)) < 1.0 / scale  # within 1 LSB


@needs_dataset
def test_speaker_pairs_are_disjoint_and_reproducible():
    pairs_a = sg.select_speaker_pairs(DATASET_DIR, 3, 20250817)
    pairs_b = sg.select_speaker_pairs(DATASET_DIR, 3, 20250817)
    assert pairs_a == pairs_b
    flat = [spk for pair in pairs_a for spk in pair]
    assert len(set(flat)) == 6  # all distinct across seeds and roles


@needs_dataset
def test_double_talk_ser_and_placement():
    cfg = _load_cfg()
    scenario = {"talk": "double", "ser_db": 0.0, "noise_type": "none",
                "snr_db": None}
    fs = cfg["sample_rate"]
    # Delayed-delta RIRs: no room sim needed to test the mixing logic.
    h_echo = np.zeros(400); h_echo[80] = 0.6
    h_near = np.zeros(400); h_near[50] = 0.8
    out = sg.synthesise(cfg, scenario, 0, h_echo, h_near)

    # Near material confined to its timeline (4–11 s, plus RIR tail).
    assert np.all(out.s_clean[: 4 * fs] == 0)
    assert np.all(out.s_clean[11 * fs :] == 0)
    # SER pinned on the overlap.
    assert abs(out.meta["ser_db_achieved"] - 0.0) < 1e-6
    overlap = out.far_mask & out.near_mask
    p_echo = np.mean(out.d_echo[overlap] ** 2)
    p_near = np.mean(out.s[overlap] ** 2)
    assert abs(10 * np.log10(p_near / p_echo)) < 0.01
    assert np.allclose(out.d, out.d_echo + out.s + out.v)


@needs_dataset
def test_noise_snr_is_achieved():
    cfg = _load_cfg()
    fs = cfg["sample_rate"]
    h_echo = np.zeros(400); h_echo[80] = 0.6
    h_near = np.zeros(400); h_near[50] = 0.8
    for noise_type in ["white", "speech_shaped"]:
        scenario = {"talk": "far_single", "ser_db": 0.0,
                    "noise_type": noise_type, "snr_db": 20.0}
        out = sg.synthesise(cfg, scenario, 1, h_echo, h_near)
        speech_mask = out.far_mask | out.near_mask
        p_speech = np.mean((out.d_echo + out.s)[speech_mask] ** 2)
        p_noise = np.mean(out.v**2)
        assert abs(10 * np.log10(p_speech / p_noise) - 20.0) < 0.01
