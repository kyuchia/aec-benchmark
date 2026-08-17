# AEC Benchmark

A reproducible benchmark comparing acoustic echo cancellation (AEC)
implementations in simulated room conditions. Systems under test: a float64
NLMS adaptive filter and the SpeexDSP MDF echo canceller, evaluated across a
controlled matrix of room acoustics, talk states, noise levels, and filter
configurations. All signals are synthesised (LibriSpeech speech material
convolved with simulated room impulse responses); no real recordings are
involved.

**Status: full T0 matrix running.** The complete two-stage experiment
matrix (Stage A: RT60 × speaker–mic distance main effects; Stage B: talk
state, background noise, tail length, and NLMS step size, one factor at a
time) runs as a single batch:

```bash
python src/run_experiment.py --batch
```

Each (scenario, seed, system) triple becomes one row in
`results/raw/runs.csv`, with metrics (segmented ERLE, convergence time,
double-talk distortion, coefficient misalignment), provenance (achieved
RT60, calibrated absorption, scaling constant, git SHA), status
(ok/diverged/failed), and wall time. Figures aggregate from that CSV only:

```bash
python -c "from pathlib import Path; import sys; sys.path.insert(0, 'src'); \
  from plotting import batch_figures; \
  batch_figures(Path('results/raw/runs.csv'), Path('results/figures'))"
```

Single-cell debug runs remain available:

```bash
python src/run_experiment.py --scenario baseline --seed 0
```

Room simulation calibrates wall absorption per RT60 level against
Schroeder-measured RT60 (with `inverse_sabine` as initialisation only);
every intermediate signal is persisted under `data/generated/`. The
report (`results/report.md`) is the remaining deliverable for this tier.

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
python scripts/fetch_data.py
```

Data lands in `data/` (gitignored, never committed).

## Tests

```bash
python -m pytest tests/
```

## Layout

```
config/scenarios.yaml   experiment matrix — single source of truth
src/                    library code and batch entry point
scripts/                one-off setup commands (data fetch)
tests/                  unit and smoke tests
data/                   fetched + generated audio (gitignored)
results/                metric CSVs, figures, report (pending)
```
