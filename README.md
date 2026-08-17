# AEC Benchmark

A reproducible benchmark comparing acoustic echo cancellation (AEC)
implementations in simulated room conditions. Systems under test: a
sample-wise float64 NLMS adaptive filter (deliberately without double-talk
protection) and the SpeexDSP MDF echo canceller (as a linear canceller, no
preprocessor), against a passthrough reference measured on the same int16
signal path. All signals are synthesised — LibriSpeech speech convolved with
image-source room impulse responses — so every echo path and near-end signal
is known exactly.

The experiment matrix crosses room reverberation (RT60 0.2–0.8 s, absorption
calibrated against Schroeder-measured RT60) with loudspeaker–microphone
distance, then varies talk state, background noise, filter tail length, and
NLMS step size one factor at a time. Metrics: activity-segmented short-time
ERLE, convergence time, double-talk near-end distortion (segmental SNR,
STOI, PESQ, log-spectral distance), and coefficient misalignment against the
true echo path.

**Findings and figures: [results/report.md](results/report.md).** Every
number in the report is rendered from `results/raw/*.csv` by script; nothing
is typed by hand.

## Setup

Requires Python 3.11+ and the SpeexDSP native library:

```bash
# macOS
brew install speexdsp
# Debian/Ubuntu
sudo apt install libspeexdsp-dev
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SpeexDSP is accessed through a thin `ctypes` binding
([src/aec_speex.py](src/aec_speex.py)) that loads the shared library from
standard install locations; set `SPEEXDSP_LIB=/path/to/libspeexdsp.{dylib,so}`
if yours is elsewhere. No compiled Python extension is required.

## Reproducing everything

```bash
python scripts/fetch_data.py          # LibriSpeech test-clean (~346 MB, md5-verified)
python -m pytest tests/               # unit + integration tests
python src/run_experiment.py --batch  # full matrix -> results/raw/runs.csv + calibration.csv
python scripts/make_figures.py        # all figures -> results/figures/
python scripts/render_report.py       # report      -> results/report.md
```

The batch (201 runs) completes in a few minutes on a laptop and is
deterministic: re-running reproduces every metric column bit-identically.
Runs that fail or diverge are recorded as rows with a `status` column and a
reason — the batch never leaves silent gaps, and the row count is asserted.

On the first batch, wall absorption is calibrated per RT60 level: pyroom-
acoustics' `inverse_sabine` provides the starting value, then bisection
adjusts absorption until the RT60 measured on the generated impulse response
(Schroeder backward integration) is within ±3% of target. Results are cached
under `data/generated/` and invalidated automatically if the room
configuration changes; provenance (Sabine vs calibrated values) is exported
to `results/raw/calibration.csv`.

Single cells can be re-run for debugging:

```bash
python src/run_experiment.py --scenario baseline --seed 0
```

## Configuration

`config/scenarios.yaml` is the single source of truth for the experiment
matrix: room geometry, levels, talk-state timelines, segmentation and metric
parameters, system configurations, and the Stage A/B factor levels. No
scenario parameter is hardcoded in Python.

## Layout

```
config/scenarios.yaml   experiment matrix — single source of truth
src/                    library code and the batch entry point
  room.py               ShoeBox RIRs, geometry checks, RT60 calibration
  signals.py            speech selection, levelling, mixing, int16 scaling
  segment.py            three-state activity segmentation
  aec_nlms.py           float64 NLMS (verified sample-exact)
  aec_speex.py          SpeexDSP ctypes binding
  metrics.py            ERLE, convergence, distortion, misalignment
  plotting.py           figure generation (reads persisted data only)
  run_experiment.py     batch driver / single-run debug entry point
scripts/                setup + rendering commands (fetch, figures, report)
tests/                  unit and integration tests
data/                   fetched + generated audio (gitignored)
results/                raw CSVs, figures, report
```
