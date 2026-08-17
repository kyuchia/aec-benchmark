"""Smoke test for the SpeexDSP binding.

Round-trips a synthetic far-end / microphone WAV pair through
run_speex_aec and checks that (a) output is well-formed and (b) the
echo is substantially attenuated once the filter has had time to adapt.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from aec_speex import run_speex_aec

SAMPLE_RATE = 16000
FRAME_SIZE = 160


def _make_wav_pair(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    n = 8 * SAMPLE_RATE  # 8 s

    x = rng.standard_normal(n)
    x /= np.max(np.abs(x))

    # Simple sparse echo path: direct path delayed 8 ms plus two reflections.
    h = np.zeros(1024)
    h[128] = 0.5
    h[300] = 0.2
    h[550] = 0.1
    d = np.convolve(x, h)[:n]

    scale = 0.5 * 32767.0
    x_path = tmp_path / "far_end.wav"
    d_path = tmp_path / "mic.wav"
    sf.write(x_path, (x * scale).astype(np.int16), SAMPLE_RATE, subtype="PCM_16")
    sf.write(d_path, (d * scale).astype(np.int16), SAMPLE_RATE, subtype="PCM_16")
    return x_path, d_path


def test_speex_reduces_echo_on_wav_pair(tmp_path):
    x_path, d_path = _make_wav_pair(tmp_path)
    x, _ = sf.read(x_path, dtype="int16")
    d, _ = sf.read(d_path, dtype="int16")

    e = run_speex_aec(x, d, FRAME_SIZE, filter_length=3200, sample_rate=SAMPLE_RATE)

    assert e.dtype == np.int16
    assert len(e) == len(d)
    assert np.any(e != 0)

    # After 4 s of adaptation the residual should be well below the echo.
    tail = slice(4 * SAMPLE_RATE, None)
    p_d = np.mean(d[tail].astype(np.float64) ** 2)
    p_e = np.mean(e[tail].astype(np.float64) ** 2)
    erle_db = 10 * np.log10(p_d / p_e)
    assert erle_db > 6.0, f"echo not reduced: tail ERLE {erle_db:.1f} dB"


def test_filter_length_must_be_frame_multiple():
    x = np.zeros(FRAME_SIZE, dtype=np.int16)
    try:
        run_speex_aec(x, x, FRAME_SIZE, filter_length=1000, sample_rate=SAMPLE_RATE)
    except ValueError:
        return
    raise AssertionError("expected ValueError for filter_length % frame_size != 0")
