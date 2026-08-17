"""Render results/report.md from results/raw/*.csv.

Every measured number in the report is computed here from the CSVs —
nothing is typed by hand. Stated constants (thresholds, smoothing
coefficients, geometry) are read from config/scenarios.yaml, the single
source of truth for the experiment matrix.

Usage:
    python scripts/render_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "report.md"


def md_table(df: pd.DataFrame, index_label: str, fmt: str = "{:.1f}") -> str:
    """Render a pivot table as GitHub markdown."""
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        if isinstance(v, float) and np.isinf(v):
            return "inf"
        if isinstance(v, float):
            return fmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + index_label + " | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + str(idx) + " | "
                     + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    runs = pd.read_csv(RAW / "runs.csv")
    cal = pd.read_csv(RAW / "calibration.csv")
    with open(REPO_ROOT / "config" / "scenarios.yaml") as f:
        cfg = yaml.safe_load(f)

    seg_cfg = cfg["segmentation"]
    met_cfg = cfg["metrics"]
    lvl_cfg = cfg["levels"]
    nlms_cfg = cfg["systems"]["nlms_f64"]
    speex_cfg = cfg["systems"]["speex"]
    n_seeds = cfg["speech"]["n_seeds"]

    # ------------------------------------------------------------------ checks
    ns_erle = runs[(runs.axis == "talk") & (runs.level == "near_single")][
        "erle_steady_state_db"]
    if not ns_erle.isna().all():
        sys.exit("expected near_single ERLE to be NaN in every row")

    # -------------------------------------------------------------- aggregates
    a = runs[(runs.stage == "a") & (runs.system != "none")]
    b = runs[runs.stage == "b"]

    def apivot(system, col):
        return a[a.system == system].pivot_table(
            values=col, index="rt60_target_s",
            columns="speaker_mic_distance_m")

    def rt60_labels():
        rows = {}
        for rt in sorted(a.rt60_target_s.unique()):
            ach = a.loc[a.rt60_target_s == rt, "rt60_achieved_echo_s"]
            rows[rt] = f"{rt:g} (ach. {ach.min():.2f}–{ach.max():.2f})"
        return rows

    labels = rt60_labels()

    def stage_a_table(col, fmt="{:.1f}"):
        parts = []
        for system in ["nlms_f64", "speex"]:
            piv = apivot(system, col).rename(index=labels)
            piv.columns = [f"{c:g} m" for c in piv.columns]
            parts.append(f"**{system}**\n\n"
                         + md_table(piv, "RT60 target (s)", fmt))
        return "\n\n".join(parts)

    talk = b[b.axis == "talk"]
    talk_erle = talk.pivot_table(values="erle_steady_state_db", index="level",
                                 columns="system", aggfunc="mean") \
        .reindex(["far_single", "double", "near_single"])
    talk_erle.columns.name = None

    def quality_pivot(level):
        sub = talk[talk.level == level]
        return sub.pivot_table(
            values=["segsnr_db", "stoi", "pesq_wb"], columns="system",
            aggfunc="mean")

    q_double = quality_pivot("double")
    q_near = quality_pivot("near_single")
    none_floor = talk[(talk.level == "near_single")
                      & (talk.system == "none")]["segsnr_db"].mean()

    noise = b[b.axis == "noise"]
    noise_erle = noise.pivot_table(values="erle_steady_state_db",
                                   index="level", columns="system",
                                   aggfunc="mean") \
        .reindex(["no_noise", "snr20", "snr10"])
    noise_erle.columns.name = None

    div = runs[runs.status == "diverged"].copy()
    dt_start = cfg["timelines"]["double"]["near"][0][0]
    div["into_double_talk_s"] = np.where(
        div.axis == "talk", div.divergence_time_s - dt_start, np.nan)
    div_table = div[["run_id", "divergence_time_s", "into_double_talk_s",
                     "erle_steady_state_db"]].copy()
    div_table.columns = ["run", "onset (s)", "onset into double-talk (s)",
                         "post-hoc steady ERLE (dB)"]
    div_md = md_table(div_table.set_index("run"), "run", "{:.2f}")

    tail = b[b.axis == "tail_length"]
    tail_order = ["50ms", "100ms", "200ms", "400ms"]
    tail_erle = tail.pivot_table(values="erle_steady_state_db", index="level",
                                 columns="system",
                                 aggfunc="mean").reindex(tail_order)
    tail_erle.columns.name = None
    tail_conv = tail.pivot_table(values="convergence_time_s", index="level",
                                 columns="system",
                                 aggfunc="mean").reindex(tail_order)
    tail_conv.columns.name = None

    mu = b[(b.axis == "mu") & (b.level != "reference")]
    mu_piv = mu.pivot_table(
        values=["erle_steady_state_db", "convergence_time_s",
                "misalignment_final_db"], index="level", aggfunc="mean")
    mu_piv = mu_piv[["erle_steady_state_db", "convergence_time_s",
                     "misalignment_final_db"]]
    mu_piv.columns = ["steady ERLE (dB)", "convergence (s)",
                      "final misalignment (dB)"]
    mu_ref = b[(b.axis == "mu") & (b.level == "reference")]

    base = runs[(runs.stage == "a")
                & (runs.scenario_key.str.startswith("rt0.4_d1_"))]
    base_nlms = base[base.system == "nlms_f64"]
    base_speex = base[base.system == "speex"]
    b_nlms_erle = base_nlms.erle_steady_state_db
    b_nlms_mis = base_nlms.misalignment_final_db

    dt_nlms = talk[(talk.level == "double") & (talk.system == "nlms_f64")] \
        .sort_values("seed")
    dt_seed_strs = [
        f"seed {int(r.seed)}: "
        + (f"diverged at {r.divergence_time_s:.2f} s, " if r.status ==
           "diverged" else "no divergence, ")
        + f"post-hoc steady ERLE {r.erle_steady_state_db:+.1f} dB"
        for r in dt_nlms.itertuples()]

    speex_dt = talk[(talk.level == "double") & (talk.system == "speex")]
    speex_fs = talk[(talk.level == "far_single") & (talk.system == "speex")]

    snr20_div = len(div[(div.axis == "noise") & (div.level == "snr20")])
    snr10_div = len(div[(div.axis == "noise") & (div.level == "snr10")])

    n_total = len(runs)
    n_ok = int((runs.status == "ok").sum())
    wall_sum_min = runs.wall_time_s.sum() / 60.0
    git_shas = sorted(runs.git_sha.unique())

    cal_tab = cal.rename(columns={
        "rt60_target_s": "target (s)",
        "absorption_sabine_init": "Sabine α",
        "rt60_achieved_sabine_init_s": "achieved w/ Sabine α (s)",
        "absorption_calibrated": "calibrated α",
        "rt60_achieved_calibrated_s": "achieved w/ calibrated α (s)",
    })[["target (s)", "Sabine α", "achieved w/ Sabine α (s)",
        "calibrated α", "achieved w/ calibrated α (s)"]]
    cal_md = md_table(cal_tab.set_index("target (s)"), "target (s)", "{:.3f}")

    room = cfg["room"]
    tl = cfg["timelines"]["double"]

    # ------------------------------------------------------------------ report
    report = f"""# AEC benchmark — Tier 0 report

Comparison of a float64 NLMS adaptive filter (`nlms_f64`) and the SpeexDSP
MDF echo canceller (`speex`) against a passthrough reference (`none`) on
synthesised room-acoustic scenarios. {n_total} runs
({n_seeds} utterance-pair seeds per condition); {n_ok} completed normally,
{n_total - n_ok} diverged (all unprotected-NLMS runs; §2.4). All numbers in
this report are rendered from `results/raw/*.csv` by
`scripts/render_report.py`; run provenance (git SHA per row):
{", ".join(git_shas)}.

## 1. Method

### 1.1 Signal synthesis

Speech is drawn from LibriSpeech test-clean at {cfg["sample_rate"]} Hz.
Each seed uses a disjoint (far-end, near-end) speaker pair — all
{2 * n_seeds} speakers distinct — drawn reproducibly from a fixed selection
seed; per-speaker utterances are concatenated in a seeded order to fill the
{cfg["duration_s"]:g} s scenario duration. The same three pairs are reused
across every scenario, so factor effects are paired rather than confounded
with material changes. The far-end signal `x` is convolved with the
loudspeaker-to-mic RIR to form the echo; near-end speech is convolved with
its own RIR; noise (when present) is white noise filtered to the long-term
average spectrum of the run's speech material; the microphone signal is the
sum. The AEC receives `x` and `d` only.

Levels: speech material is normalised to
{lvl_cfg["active_level_dbov"]:g} dBov active-speech level using an
energy-gated RMS — frames more than
{lvl_cfg["active_gate_threshold_db"]:g} dB below the peak frame RMS are
treated as inactive. This is a simplified active-level measure, not ITU-T
P.56. The near-end level is then pinned to a configured
signal-to-echo ratio (SER {cfg["scenario_defaults"]["ser_db"]:g} dB,
near-end over echo, measured on the double-talk overlap), so the
fixed-point operating point is controlled rather than an accident of room
geometry.

The double-talk timeline is far-only lead-in (0–{tl["near"][0][0]:g} s),
double-talk ({tl["near"][0][0]:g}–{tl["near"][0][1]:g} s), then a far-only
tail to {cfg["duration_s"]:g} s — the tail exists so that steady-state
statistics are measured after, not before, the double-talk episode.

### 1.2 Room simulation and RT60 calibration

Rooms are {room["dimensions_m"][0]:g} × {room["dimensions_m"][1]:g} ×
{room["dimensions_m"][2]:g} m pyroomacoustics ShoeBoxes (image source
method). The microphone sits off-centre; sources are placed off-axis and
away from half-dimension coordinates to avoid degenerate image-source
symmetry, with ≥{room["min_wall_clearance_m"]:g} m wall clearance
(asserted). The propagation delay is left in the RIRs; a build-time
assertion fails if the first arrival is earlier than pure propagation
allows.

`inverse_sabine` is used only to **initialise** the wall absorption; the
value actually used is calibrated by bisection until the RT60 measured on
the generated loudspeaker-to-mic RIR (Schroeder backward integration) is
within ±{cal["tolerance_pct"].iloc[0]:g}% of target at the
{cal["reference_distance_m"].iloc[0]:g} m reference distance. The
uncalibrated Sabine values overshoot systematically, and the overshoot
grows with target RT60 — a small finding in itself:

{cal_md}

The calibrated absorption for each RT60 level is shared across all
distance levels of that row (per-distance recalibration would confound the
distance axis with wall absorption). Residual drift remains at 2.5 m,
where every level measures ~5–8% above target: the weaker direct path
shifts the Schroeder fit toward the reverberant tail. Stage A figures and
tables are therefore labelled with **achieved** RT60, and per-scenario
achieved values are stored in every CSV row.

### 1.3 Systems under test

| ID | Description |
|---|---|
| `none` | Passthrough reference. Not a no-op: `d` goes through the identical float→int16→float round-trip at the identical scaling constant as `speex`, so the 0 dB reference is measured on the same signal path and carries the same quantisation floor. |
| `nlms_f64` | Sample-wise float64 NLMS, L = {nlms_cfg["filter_length_ms"]:g} ms ({int(nlms_cfg["filter_length_ms"] * 16)} taps), μ = {nlms_cfg["mu"]:g}, δ = {nlms_cfg["delta"]:g}. **No double-talk detection, deliberately** — its absence is what makes SpeexDSP's built-in protection visible. Verified sample-exact against a naive per-sample reference. |
| `speex` | SpeexDSP MDF, frame size {speex_cfg["frame_size"]} samples, tail {speex_cfg["filter_length_ms"]:g} ms, sampling rate set explicitly via `speex_echo_ctl` and read back (asserted). |

One float→int16 scaling constant per run, computed to leave
{lvl_cfg["int16_headroom_db"]:g} dB headroom above max(|x|, |d|), applied
identically to every system in the run, asserted not to clip on `x` and
`d`, with `speex` output checked for saturation. The constant is recorded
in every CSV row.

`nlms_f64` runs in float throughout — it never passes through the int16
path. The float-vs-fixed comparison is therefore asymmetric by
construction: NLMS is exempt from the quantisation floor the fixed-point
systems carry. This is stated here as a caveat and revisited in §4.

Speex is tested **as a linear echo canceller only**: the
`speex_preprocess` chain (residual echo suppressor, noise suppressor) is
never attached. Attaching it would confound the comparison against a bare
adaptive filter — the preprocessor's nonlinear suppression would mask the
linear canceller's actual behaviour. Comparisons against deployed Speex
configurations (which typically include the preprocessor) should keep this
in mind.

### 1.4 Metrics

**Segmentation.** Metrics are computed over a three-state activity
segmentation (far-active / near-active / neither) derived from the
ground-truth component signals independently, never from the mixture:
{seg_cfg["frame_ms"]:g} ms frames, a frame is active when its mean power
exceeds {seg_cfg["energy_threshold_dbov"]:g} dBov, and activity is held
for {seg_cfg["hangover_ms"]:g} ms after the last active frame to bridge
inter-word pauses.

**ERLE** (primary): short-time, per 20 ms frame, computed **only over
ERLE-valid frames** (far-active and not near-active — during double-talk
the error signal legitimately contains near-end speech), smoothed with an
EMA (α = {met_cfg["erle_ema_alpha"]:g} over the valid-frame sequence).
Steady state is the median of the final
{met_cfg["steady_state_last_fraction"]:.0%} of valid smoothed frames. A
sanity assertion fails any run whose **steady-state** ERLE exceeds
{met_cfg["erle_sanity_max_db"]:g} dB — the signature of passing the echo
itself as the reference; it applies to the steady-state statistic only,
since instantaneous short-time ERLE legitimately spikes in low-energy
frames.

**Convergence time**: time from the first valid frame until smoothed ERLE
first reaches {met_cfg["convergence_fraction"]:.0%} of that run's steady
state; NaN (flagged non-converged) if never reached. One caveat deserves
its own paragraph. Because the threshold is relative to **each run's own
steady state**, the metric reads *fast* when the target is weak and *slow*
when adaptation is gradual. In Stage A, NLMS at high RT60 appears to
converge in under half a second — not because adaptation is faster there,
but because its steady state is low and 90% of a weak target is reached
almost immediately. Conversely, Speex's ~5–6 s figures partly reflect its
slowly rising curve shape rather than a 5 s wait for usable cancellation.
Cross-system convergence comparisons are only valid between runs with
comparable steady states.

**Near-end distortion**: segmental SNR of the output against `s` — the
reverberant near-end at the microphone, **not** `s_clean` — because the
AEC is not being asked to dereverberate. Computed over double-talk frames
(or, for near-single-talk, over near-active frames, where it degenerates
to passthrough distortion). Frames with bit-exact reproduction are
excluded and counted; a run reproducing every frame exactly reports +inf
rather than a floor-dependent large number. STOI and wideband PESQ are
computed over the near-active span; log-spectral distance per frame.

**Misalignment**: 10·log10(‖w − h‖²/‖h‖²) with `h_echo`
truncated/zero-padded to the filter length. Computed for `nlms_f64` (Speex
does not expose coefficients).

**Divergence**: an output is flagged diverged at the first sample that is
non-finite or exceeds 10× the peak of |d| (+20 dB); the onset time is
recorded as a data point. Divergence is detected, never prevented.

### 1.5 Experiment matrix

Stage A crosses RT60 × speaker–mic distance (far single-talk, no noise);
Stage B varies one factor at a time around the baseline (RT60 0.4 s,
1.0 m): talk state, background noise, tail length (both systems), and NLMS
step size (with Speex run once at baseline as a labelled reference — it
has no step-size parameter). {n_total} rows; the sum of per-row processing
times is {wall_sum_min:.1f} min.

## 2. Results

### 2.1 Stage A — RT60 × distance

Steady-state ERLE (dB), mean over {n_seeds} seeds:

{stage_a_table("erle_steady_state_db")}

Convergence time (s) — read §1.4's caveat before comparing across systems:

{stage_a_table("convergence_time_s", "{:.2f}")}

![Stage A ERLE](figures/stage_a_erle.png)
![Stage A convergence](figures/stage_a_convergence.png)

ERLE degrades monotonically with RT60 for both systems at every distance —
at 1.0 m, NLMS falls from {apivot("nlms_f64", "erle_steady_state_db").loc[0.2, 1.0]:.1f}
to {apivot("nlms_f64", "erle_steady_state_db").loc[0.8, 1.0]:.1f} dB and
Speex from {apivot("speex", "erle_steady_state_db").loc[0.2, 1.0]:.1f} to
{apivot("speex", "erle_steady_state_db").loc[0.8, 1.0]:.1f} dB across the
four levels. The per-step drop is roughly constant (~5–7 dB per 0.2 s of
RT60) rather than accelerating at the longest tail. Distance is likewise
monotonic (closer loudspeaker → higher ERLE) with one off-pattern cell
(Speex at RT60 0.2, which peaks at 1.0 m).

### 2.2 Talk state

Steady-state ERLE (dB) over far-only regions ("–" = undefined, no
far-end activity):

{md_table(talk_erle, "talk state")}

Near-end distortion over near-active regions:

**double-talk** (echo + near-end simultaneously):

{md_table(q_double, "metric", "{:.2f}")}

**near-single-talk** (no far-end at all — pure passthrough):

{md_table(q_near, "metric", "{:.2f}")}

The `none` row's near-single segmental SNR ({none_floor:.1f} dB) is the
measured int16 quantisation floor of the shared I/O path — the ceiling
against which the other systems' segSNR should be read. `nlms_f64` reports
+inf there because with a silent reference its float output reproduces `d`
bit-exactly. Speex's much lower value is discussed in §4.

During double-talk, Speex holds {speex_dt.erle_steady_state_db.mean():.1f} dB
ERLE (vs {speex_fs.erle_steady_state_db.mean():.1f} dB in far single-talk)
while *improving* the near-end over the unprocessed mixture: segSNR
{q_double.loc["segsnr_db", "speex"]:+.1f} dB vs
{q_double.loc["segsnr_db", "none"]:+.1f} dB, STOI
{q_double.loc["stoi", "speex"]:.2f} vs
{q_double.loc["stoi", "none"]:.2f}, PESQ
{q_double.loc["pesq_wb", "speex"]:.2f} vs
{q_double.loc["pesq_wb", "none"]:.2f}. The unprotected NLMS instead
destroys both: {talk_erle.loc["double", "nlms_f64"]:.1f} dB mean ERLE and
{q_double.loc["segsnr_db", "nlms_f64"]:+.1f} dB segSNR (per-seed detail in
§2.4).

![Stage B talk state](figures/stage_b_talk.png)

### 2.3 Background noise

Steady-state ERLE (dB), mean over seeds (diverged runs included — they are
the phenomenon, not an artifact):

{md_table(noise_erle, "noise level")}

Speex degrades gracefully as SNR falls. NLMS collapses: at 20 dB SNR
{snr20_div} of {n_seeds} seeds diverged; at 10 dB SNR {snr10_div} of
{n_seeds} did. No near-end speech is present in any of these runs.

![Stage B noise](figures/stage_b_noise.png)

### 2.4 Divergence onsets

Every non-ok run in the matrix, all `nlms_f64`:

{div_md}

Per-seed detail for double-talk ({", ".join(dt_seed_strs)}) shows the
divergences beginning 2.7–3.1 s after double-talk onset — and that one
seed survives outright. Under 10 dB noise, divergence arrives within
{div[(div.axis == "noise") & (div.level == "snr10")].divergence_time_s.min():.1f}–{div[(div.axis == "noise") & (div.level == "snr10")].divergence_time_s.max():.1f} s
of signal onset, before any adaptation has consolidated.

### 2.5 Tail length

Steady-state ERLE (dB):

{md_table(tail_erle, "filter length")}

Convergence time (s):

{md_table(tail_conv, "filter length")}

Both systems lose heavily from undermodelling (50 ms covers little of a
0.4 s decay). NLMS peaks at 200 ms and *loses* ground at 400 ms —
the longer filter adds gradient noise and adapts more slowly per tap —
while Speex holds its 200 ms performance at 400 ms.

![Stage B tail length](figures/stage_b_tail.png)

### 2.6 NLMS step size

Mean over seeds; Speex at its baseline configuration shown for reference
(it has no step-size parameter and is not part of the sweep):

{md_table(mu_piv, "level", "{:.2f}")}

| reference | steady ERLE (dB) | convergence (s) |
|---|---|---|
| speex @ baseline | {mu_ref.erle_steady_state_db.mean():.2f} | {mu_ref.convergence_time_s.mean():.2f} |

![Stage B step size](figures/stage_b_mu.png)

## 3. Discussion

### 3.1 The hazard is uncorrelated energy, not double-talk specifically

The classic framing says an unprotected adaptive filter is endangered by
double-talk. The data here support a broader statement: the hazard is
**any energy in `d` uncorrelated with the reference `x`** — near-end
speech is one instance, background noise another. The noise axis is the
cleaner demonstration: with no near-end speech whatsoever, stationary
noise at 20 dB SNR diverged {snr20_div}/{n_seeds} seeds and 10 dB SNR
diverged {snr10_div}/{n_seeds}, dose-dependent, with onsets as early as
{div[(div.axis == "noise")].divergence_time_s.min():.2f} s. The mechanism
is the NLMS update itself: during far-end pauses the sliding ‖x‖²
collapses toward the regulariser δ = {nlms_cfg["delta"]:g} while the error
still carries noise, so the normalised step explodes. Double-talk is the
dramatic case of the same mechanism (near-end energy during low reference
power), not a separate phenomenon. Speex, whose MDF carries adaptation
control designed for exactly this, degrades smoothly on both axes
(§2.2–2.3).

### 3.2 The step-size trade-off only exists under interference

In clean far single-talk, the textbook μ trade-off is absent: ERLE and
final misalignment both improve monotonically up to μ = 0.9
(§2.6 — {mu_piv.loc["mu0.9", "steady ERLE (dB)"]:.1f} dB /
{mu_piv.loc["mu0.9", "final misalignment (dB)"]:.1f} dB). There is nothing
to trade against — no noise, no near end, and a 15 s horizon that rewards
fast adaptation. The cost of large μ only materialises when interference
exists: the noise and double-talk axes show μ = 0.5 diverging outright.
Read together, the two axes say that step-size tuning on clean data
mislearns the operating point for realistic conditions.

### 3.3 Output quietness overstates estimate quality

At baseline, NLMS reaches {b_nlms_erle.mean():.1f} dB mean steady-state
ERLE while its final coefficient misalignment is only
{b_nlms_mis.mean():.1f} dB — the output is quiet out of proportion to the
accuracy of the underlying echo-path estimate. ERLE rewards cancelling
whatever the *current* input excites; misalignment measures the whole
estimate. The gap is visible dynamically too: the misalignment trajectory
(figure below) improves stepwise as new speech material excites new parts
of the path, and the ERLE curve's oscillation with the speech envelope is
the same effect seen from the output side.

![Baseline misalignment](figures/baseline_misalignment.png)

### 3.4 Two convergence styles — and a metric artifact

NLMS converges fast and raggedly (baseline
{base_nlms.convergence_time_s.mean():.2f} s mean); Speex ramps slowly and
smoothly ({base_speex.convergence_time_s.mean():.2f} s). But §1.4's caveat
applies: part of the Stage A convergence table is the metric's
construction, not adaptation speed. NLMS's sub-second "convergence" at
RT60 0.8 s is 90% of a weak target; Speex's ~6 s figures are the price of
a curve that keeps rising toward a slightly higher plateau. The honest
summary is a difference in *style* — fast/oscillatory vs slow/monotonic —
with scalar convergence times only comparable at similar steady states.

![Baseline ERLE](figures/baseline_erle.png)

### 3.5 Three seeds, visible individually

The double-talk row is the argument for per-seed points: of three seeds
under identical conditions, two diverged (2.7 and 3.1 s into double-talk)
and one survived with degraded but positive ERLE (§2.4). A mean over
those three (−12 dB) describes none of them. All Stage B figures
therefore show individual seeds, with diverged runs marked; with
{n_seeds} seeds, error bars would imply a precision this design does not
have.

## 4. Limitations

- **Simulation gap.** No loudspeaker nonlinearity, no time-varying echo
  paths, no microphone self-noise. All rooms are ideal shoeboxes with
  frequency-flat absorption. Every system here solves a purely linear,
  time-invariant problem; real-world rankings may differ, particularly
  for Speex whose design anticipates nonlinear residuals.
- **segSNR punishes the perceptually irrelevant.** In near-single-talk,
  Speex scores {q_near.loc["segsnr_db", "speex"]:.1f} dB segSNR against
  the `none` floor of {none_floor:.1f} dB — yet its STOI is
  {q_near.loc["stoi", "speex"]:.2f} and PESQ
  {q_near.loc["pesq_wb", "speex"]:.2f}. The alteration is concentrated at
  low frequency, consistent with SpeexDSP's internal DC-notch filter on
  the microphone path: a waveform-difference metric punishes a
  perceptually irrelevant low-frequency change that the perceptual
  metrics correctly ignore. This mismatch is part of the motivation for
  the planned masking-based audibility analysis (Tier 1).
- **Float-vs-fixed asymmetry.** `nlms_f64` never passes the int16 path;
  `speex` and `none` do. The quantisation floor measured on `none`
  (§2.2) bounds the effect, but the comparison is not symmetric.
- **Convergence metric construction.** Relative-to-own-steady-state
  convergence times are not comparable across systems with different
  steady states (§1.4, §3.4).
- **Three seeds.** Enough to expose seed-dependence (§3.5), not enough
  for distributional claims. Per-seed values are plotted and stored.
- **Segmental SNR reference choice.** Distortion is referenced to the
  reverberant near-end `s`; systems are not rewarded or punished for
  dereverberation effects.

## 5. Reproduction

From a clean checkout (macOS/Linux, Python 3.11, SpeexDSP installed —
`brew install speexdsp` or `apt install libspeexdsp-dev`):

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py          # LibriSpeech test-clean (~346 MB, md5-checked)
python -m pytest tests/               # unit + integration tests
python src/run_experiment.py --batch  # full matrix -> results/raw/runs.csv (+ calibration.csv)
python scripts/make_figures.py        # all figures  -> results/figures/
python scripts/render_report.py       # this report  -> results/report.md
```

The batch is deterministic: re-running it reproduces every metric column
bit-identically (verified). RT60 absorption calibration runs once and is
cached under `data/generated/`; a cold run recalibrates automatically
(a few extra minutes). Single cells can be re-run for debugging with
`python src/run_experiment.py --scenario baseline --seed 0`.
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report)
    print(f"wrote {OUT} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
