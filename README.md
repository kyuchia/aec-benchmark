# AEC Benchmark

A reproducible benchmark comparing acoustic echo cancellation (AEC)
implementations in simulated room conditions. Systems under test: a float64
NLMS adaptive filter and the SpeexDSP MDF echo canceller, evaluated across a
controlled matrix of room acoustics, talk states, noise levels, and filter
configurations. All signals are synthesised (LibriSpeech speech material
convolved with simulated room impulse responses); no real recordings are
involved.

**Status: early setup.** Environment, SpeexDSP binding, and speech-data
fetching are in place. Scenario simulation, the AEC implementations under a
common interface, metrics, and the experiment matrix are being added
incrementally; results will live in `results/`.

## Setup

Requires Python 3.11+ and the SpeexDSP native library:

```bash
# macOS
brew install speexdsp
# Debian/Ubuntu
sudo apt install libspeexdsp-dev
```

Then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SpeexDSP is accessed through a thin `ctypes` binding
([src/aec_speex.py](src/aec_speex.py)) that loads the shared library from
standard install locations; set `SPEEXDSP_LIB=/path/to/libspeexdsp.{dylib,so}`
if yours is elsewhere. No compiled Python extension is required.

## Speech data

Speech material is the LibriSpeech `test-clean` subset (~346 MB download,
md5-verified):

```bash
python src/fetch_data.py
```

Data lands in `data/` (gitignored, never committed).

## Tests

```bash
python -m pytest tests/
```

## Layout

```
config/scenarios.yaml   experiment matrix — single source of truth (pending)
src/                    library code and batch entry point
tests/                  unit and smoke tests
data/                   fetched + generated audio (gitignored)
results/                metric CSVs, figures, report (pending)
```
