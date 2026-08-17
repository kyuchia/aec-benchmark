"""SpeexDSP MDF echo canceller, wrapped via a thin ctypes binding.

Only the four symbols the benchmark needs are bound:

    speex_echo_state_init
    speex_echo_cancellation
    speex_echo_state_destroy
    speex_echo_ctl

The library is located from, in order: the SPEEXDSP_LIB environment
variable, ctypes.util.find_library, and a list of conventional install
paths (Homebrew on macOS, system lib dirs on Linux).

I/O is int16 throughout, matching the underlying C API. Float-to-int16
scaling is the caller's responsibility (see signals.py); this module
deliberately does not rescale.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import numpy as np

# From speex/speex_echo.h
SPEEX_ECHO_GET_FRAME_SIZE = 3
SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_ECHO_GET_SAMPLING_RATE = 25

_CANDIDATE_PATHS = [
    "/opt/homebrew/lib/libspeexdsp.dylib",
    "/usr/local/lib/libspeexdsp.dylib",
    "/usr/lib/libspeexdsp.so.1",
    "/usr/lib/x86_64-linux-gnu/libspeexdsp.so.1",
    "/usr/lib/aarch64-linux-gnu/libspeexdsp.so.1",
]


def _load_libspeexdsp() -> ctypes.CDLL:
    candidates = []
    env_path = os.environ.get("SPEEXDSP_LIB")
    if env_path:
        candidates.append(env_path)
    found = ctypes.util.find_library("speexdsp")
    if found:
        candidates.append(found)
    candidates.extend(_CANDIDATE_PATHS)

    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise OSError(
        "libspeexdsp not found. Install it (macOS: `brew install speexdsp`, "
        "Debian/Ubuntu: `sudo apt install libspeexdsp-dev`) or point the "
        "SPEEXDSP_LIB environment variable at the shared library."
    )


_lib = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = _load_libspeexdsp()
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_cancellation.restype = None
        lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
        ]
        lib.speex_echo_state_destroy.restype = None
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_echo_ctl.restype = ctypes.c_int
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _lib = lib
    return _lib


class SpeexEchoCanceller:
    """Stateful wrapper around one SpeexEchoState.

    Parameters
    ----------
    frame_size : samples per processing frame (e.g. 160 at 16 kHz = 10 ms).
    filter_length : echo tail length in samples; must be a multiple of
        frame_size.
    sample_rate : set explicitly via speex_echo_ctl and read back to verify
        it was applied — Speex's internal default is otherwise wrong for
        16 kHz material.
    """

    def __init__(self, frame_size: int, filter_length: int, sample_rate: int):
        if filter_length % frame_size != 0:
            raise ValueError(
                f"filter_length ({filter_length}) must be a multiple of "
                f"frame_size ({frame_size})"
            )
        self._lib = _get_lib()
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate

        self._state = self._lib.speex_echo_state_init(frame_size, filter_length)
        if not self._state:
            raise RuntimeError("speex_echo_state_init returned NULL")

        rate_in = ctypes.c_int(sample_rate)
        self._lib.speex_echo_ctl(
            self._state, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate_in)
        )
        rate_out = ctypes.c_int(0)
        self._lib.speex_echo_ctl(
            self._state, SPEEX_ECHO_GET_SAMPLING_RATE, ctypes.byref(rate_out)
        )
        if rate_out.value != sample_rate:
            self.close()
            raise RuntimeError(
                f"SPEEX_ECHO_SET_SAMPLING_RATE not applied: requested "
                f"{sample_rate}, state reports {rate_out.value}"
            )

    def process_frame(self, rec: np.ndarray, play: np.ndarray) -> np.ndarray:
        """Cancel one frame. rec = microphone, play = far-end reference."""
        if rec.dtype != np.int16 or play.dtype != np.int16:
            raise TypeError("rec and play must be int16")
        if len(rec) != self.frame_size or len(play) != self.frame_size:
            raise ValueError(f"frames must be exactly {self.frame_size} samples")
        out = np.empty(self.frame_size, dtype=np.int16)
        self._lib.speex_echo_cancellation(
            self._state,
            rec.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            play.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
        return out

    def close(self) -> None:
        if getattr(self, "_state", None):
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None

    def __enter__(self) -> "SpeexEchoCanceller":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def run_speex_aec(
    x: np.ndarray,
    d: np.ndarray,
    frame_size: int,
    filter_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Run the Speex canceller over full int16 signals.

    x : far-end reference (loudspeaker signal), int16
    d : microphone signal, int16
    Returns e, the echo-cancelled output, same length as d. The trailing
    partial frame, if any, is zero-padded on input and trimmed on output.
    """
    if x.dtype != np.int16 or d.dtype != np.int16:
        raise TypeError("x and d must be int16 (scaling happens upstream)")
    if len(x) != len(d):
        raise ValueError(f"length mismatch: len(x)={len(x)}, len(d)={len(d)}")

    n = len(d)
    n_padded = ((n + frame_size - 1) // frame_size) * frame_size
    x_p = np.zeros(n_padded, dtype=np.int16)
    d_p = np.zeros(n_padded, dtype=np.int16)
    x_p[:n] = x
    d_p[:n] = d

    e = np.empty(n_padded, dtype=np.int16)
    with SpeexEchoCanceller(frame_size, filter_length, sample_rate) as aec:
        for start in range(0, n_padded, frame_size):
            stop = start + frame_size
            e[start:stop] = aec.process_frame(d_p[start:stop], x_p[start:stop])
    return e[:n]
