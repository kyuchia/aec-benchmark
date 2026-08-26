# AEC Benchmark

A reproducible benchmark of acoustic echo cancellation under controlled,
simulated room conditions. This repository implements the float64 NLMS
canceller, the Q15 fixed-point NLMS (saturating arithmetic with stall and
saturation instrumentation), the experimental framework, RT60 calibration,
the evaluation logic, and the masking-based audibility analysis. SpeexDSP's
MDF canceller is used and tested through a thin `ctypes` binding — it is not
implemented here — and room impulse responses are generated using
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics); no room simulator
was written from scratch.

Microphone mixtures are synthesised from LibriSpeech speech and simulated
room impulse responses, so every echo path, near-end signal, and noise
component is known exactly — which makes exact residual-echo isolation and
coefficient-misalignment measurement possible.

## Key results

Steady-state ERLE, mean over 3 utterance-pair seeds:

| Condition       | Float NLMS | Q15 NLMS | SpeexDSP |
| --------------- | ---------: | -------: | -------: |
| Clean baseline  |    26.5 dB |  19.8 dB |  26.8 dB |
| Double-talk     |   -12.1 dB |  10.8 dB |  24.4 dB |
| 20 dB SNR noise |    -1.8 dB |   9.6 dB |  15.0 dB |
| 10 dB SNR noise |   -10.7 dB |   5.3 dB |   7.4 dB |

**Diverged runs:** the unprotected Float NLMS diverged in 2/3 double-talk
runs, 1/3 runs at 20 dB SNR, and 3/3 runs at 10 dB SNR.

1. Float NLMS and SpeexDSP reach similar ERLE at the clean baseline.
2. SpeexDSP stays useful under double-talk and background noise, while the
   deliberately unprotected Float NLMS can diverge outright.
3. Saturating Q15 arithmetic converts the same catastrophic float failure
   into bounded degradation — bounded, not healthy: the contained runs still
   leak echo and end far from the true echo path.
4. Reduced coefficient precision fails sharply under the floor-masked
   storage convention tested here (11 bits is already worse than no
   processing); this does not generalise to all fixed-point rounding
   schemes.
5. At the clean baseline, SpeexDSP and Float NLMS have nearly equal ERLE
   (26.8 vs 26.5 dB) but different residual audible fractions (0.83 vs
   0.91) under the same masking model — equal energy suppression,
   differently distributed residual.
6. At a 200 ms tail, derived cost is 369 MAC/sample for SpeexDSP's
   partitioned frequency-domain structure vs 6,400 MAC/sample for
   time-domain NLMS.

![Echo suppression under background noise](results/figures/stage_b_noise.png)

## Benchmark

279 runs over 3 reproducibly selected utterance-pair seeds: RT60 0.2–0.8 s
crossed with loudspeaker–microphone distance, then one-factor-at-a-time
sweeps of talk state, background noise, filter tail length, NLMS step size,
and Q15 coefficient precision. Evaluation covers activity-segmented ERLE,
near-end quality during double-talk (segmental SNR, STOI, PESQ), coefficient
misalignment against the true echo path, fixed-point stall and saturation
events, masking-based residual-echo audibility, and computational cost
(measured real-time factor plus analytically derived MAC counts and state
sizes).

Full methods, caveats, figures, and analysis: [results/report.md](results/report.md).

## Reproduce

Requires Python 3.11+ and the SpeexDSP native library
(`brew install speexdsp` on macOS, `sudo apt install libspeexdsp-dev` on
Debian/Ubuntu; set `SPEEXDSP_LIB=/path/to/libspeexdsp.{dylib,so}` if it is
installed elsewhere).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py          # LibriSpeech test-clean (~346 MB, md5-verified)
python -m pytest tests/               # unit + integration tests
python src/run_experiment.py --batch  # full matrix -> results/raw/runs.csv + calibration.csv
python scripts/measure_cost.py        # canceller-only RTF + derived cost -> results/raw/cost.csv
python scripts/make_figures.py        # all figures -> results/figures/
python scripts/render_report.py       # report      -> results/report.md
```

The batch (279 runs) completes in roughly 20 minutes on a laptop and is
deterministic: re-running reproduces every metric column bit-identically.
`config/scenarios.yaml` is the single source of truth for the experiment
matrix, and every number in the report is rendered from the persisted
results in `results/raw/` rather than typed by hand.
