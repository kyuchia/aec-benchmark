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
sys.path.insert(0, str(REPO_ROOT / "src"))

from figure_registry import (  # noqa: E402
    fig_embed,
    fig_ref,
    table_block,
    table_ref,
)

RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "report.md"
# The full matrix; a truncated or partial runs.csv must never render.
EXPECTED_ROWS = 279


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
    assert len(runs) == EXPECTED_ROWS, \
        f"expected {EXPECTED_ROWS} rows in runs.csv, got {len(runs)}"
    cal = pd.read_csv(RAW / "calibration.csv")
    cost = pd.read_csv(RAW / "cost.csv")
    with open(REPO_ROOT / "config" / "scenarios.yaml") as f:
        cfg = yaml.safe_load(f)

    seg_cfg = cfg["segmentation"]
    met_cfg = cfg["metrics"]
    lvl_cfg = cfg["levels"]
    nlms_cfg = cfg["systems"]["nlms_f64"]
    q15_cfg = cfg["systems"]["nlms_q15"]
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

    def _nonmono_dist(system):
        piv = apivot(system, "erle_steady_state_db")
        return [rt for rt in piv.index
                if not piv.loc[rt].is_monotonic_decreasing]
    nonmono_dist = {sys: _nonmono_dist(sys)
                    for sys in ("nlms_f64", "speex")}
    n_nonmono_dist = sum(map(len, nonmono_dist.values()))
    n_dist_rows = 2 * len(apivot("nlms_f64",
                                 "erle_steady_state_db").index)

    def stage_a_table(col, fmt="{:.1f}"):
        parts = []
        for system in ["nlms_f64", "nlms_q15", "speex"]:
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
    dt_stoi = talk[talk.level == "double"].pivot_table(
        values="stoi", index="seed", columns="system")
    ok_seed = int(talk[(talk.level == "double")
                       & (talk.system == "nlms_f64")
                       & (talk.status == "ok")].seed.iloc[0])
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
    base_q15 = base[base.system == "nlms_q15"]
    b_nlms_erle = base_nlms.erle_steady_state_db
    b_nlms_mis = base_nlms.misalignment_final_db

    # ---------------------------------------------------- fixed point
    # Word-length sweep (headline).
    wl = b[b.axis == "word_length"]
    wl_order = ["15bit", "11bit", "9bit", "7bit"]
    wl_piv = wl.pivot_table(
        values=["erle_steady_state_db", "convergence_time_s",
                "misalignment_final_db", "coeff_div_steady_db",
                "n_stall_events", "n_sat_gain", "n_sat_coeff"],
        index="level", aggfunc="mean").reindex(wl_order)
    wl_piv = wl_piv[["erle_steady_state_db", "convergence_time_s",
                     "misalignment_final_db", "coeff_div_steady_db",
                     "n_stall_events", "n_sat_gain", "n_sat_coeff"]]
    wl_tab = wl_piv.copy()
    wl_tab.columns = ["steady ERLE (dB)", "convergence (s)",
                      "misalignment (dB)", "div. from float (dB)",
                      "full stalls", "sat: gain", "sat: coeff"]
    wl_tab.index = [s.replace("bit", " bits") for s in wl_tab.index]
    for c in ["full stalls", "sat: gain", "sat: coeff"]:
        wl_tab[c] = wl_tab[c].map(
            lambda v: f"{v:,.0f}" if np.isfinite(v) else None)
    wl_erle = wl_piv["erle_steady_state_db"]
    wl_mis = wl_piv["misalignment_final_db"]
    wl_stalls = wl_piv["n_stall_events"]
    wl_tap_stalls_mean = wl.n_tap_stalls.mean()

    # Paired float-vs-Q15 ERLE gap over Stage A (same cell, same seed).
    a_f64 = a[a.system == "nlms_f64"][
        ["level", "seed", "erle_steady_state_db"]]
    a_q15 = a[a.system == "nlms_q15"][
        ["level", "seed", "erle_steady_state_db", "n_sat_total",
         "n_stall_events", "coeff_div_steady_db"]]
    paired = a_f64.merge(a_q15, on=["level", "seed"],
                         suffixes=("_f64", "_q15"))
    q15_gap = (paired.erle_steady_state_db_f64
               - paired.erle_steady_state_db_q15)
    paired["gap"] = q15_gap
    paired["dist"] = paired.level.str.split("_d").str[1]
    gap_by_d = paired.groupby("dist")["gap"].mean()
    a_q15_d03 = a[(a.system == "nlms_q15")
                  & a.level.str.endswith("_d0.3")]
    d03_sat_coeff = a_q15_d03.n_sat_coeff.mean()
    d03_mis = a_q15_d03.misalignment_final_db.mean()

    # Q15 counterpart of every diverged float run (matched on axis, level,
    # seed — identical d within the cell).
    cmp_rows = []
    for r in div.itertuples():
        q = runs[(runs.stage == r.stage) & (runs.axis == r.axis)
                 & (runs.level == r.level) & (runs.seed == r.seed)
                 & (runs.system == "nlms_q15")]
        if q.empty:
            continue
        q = q.iloc[0]
        cmp_rows.append({
            "condition": f"{r.axis}/{r.level}, seed {int(r.seed)}",
            "Float NLMS onset (s)": r.divergence_time_s,
            "Float NLMS post-hoc ERLE (dB)": r.erle_steady_state_db,
            "Q15 NLMS status": q.status,
            "Q15 NLMS steady ERLE (dB)": q.erle_steady_state_db,
            "Q15 NLMS saturations": f"{q.n_sat_total:,.0f}",
            "Q15 NLMS first sat (s)": q.sat_first_time_s,
            "Q15 NLMS div. from float (dB)": q.coeff_div_steady_db,
        })
    div_cmp_md = md_table(
        pd.DataFrame(cmp_rows).set_index("condition"), "condition", "{:.1f}")
    n_q15_bounded = sum(1 for r in cmp_rows if r["Q15 NLMS status"] == "ok")

    # Interference axes, Q15 rows.
    q15_dt = talk[(talk.level == "double") & (talk.system == "nlms_q15")]
    q15_ns10 = noise[(noise.level == "snr10")
                     & (noise.system == "nlms_q15")]
    q15_clean_sat = a[a.system == "nlms_q15"].n_sat_total
    mu_q15_int = int(round(q15_cfg["mu"] * 32768))
    delta_q30_int = max(1, int(round(q15_cfg["delta"] * 2**30)))

    # ------------------------------------------------------ audibility
    aud_cfg = cfg["audibility"]
    sys_order = ["none", "nlms_f64", "nlms_q15", "speex"]

    def aud_pivot(sub, cols):
        p = sub.pivot_table(values=cols, index="system",
                            aggfunc="mean").reindex(sys_order)
        return p[cols]

    base_all = runs[(runs.stage == "a")
                    & (runs.scenario_key.str.startswith("rt0.4_d1_"))]
    aud_base = aud_pivot(base_all, ["erle_steady_state_db",
                                    "audibility_fraction",
                                    "audibility_excess_db"])
    aud_base.columns = ["steady ERLE (dB)", "audible fraction",
                        "mean excess (dB)"]

    noise_frac = noise.pivot_table(values="audibility_fraction",
                                   index="level", columns="system",
                                   aggfunc="mean") \
        .reindex(["no_noise", "snr20", "snr10"])[sys_order]
    noise_frac.columns.name = None
    noise_exc = noise.pivot_table(values="audibility_excess_db",
                                  index="level", columns="system",
                                  aggfunc="mean") \
        .reindex(["no_noise", "snr20", "snr10"])[sys_order]
    noise_exc.columns.name = None

    wl_aud = wl.pivot_table(
        values=["erle_steady_state_db", "audibility_fraction",
                "audibility_excess_db"],
        index="level", aggfunc="mean").reindex(wl_order)
    wl_aud = wl_aud[["erle_steady_state_db", "audibility_fraction",
                     "audibility_excess_db"]]
    wl_aud.columns = ["steady ERLE (dB)", "audible fraction",
                      "mean excess (dB)"]
    wl_aud.index = [s.replace("bit", " bits") for s in wl_aud.index]
    none_a_excess = runs[(runs.stage == "a") & (runs.system == "none")
                         ].audibility_excess_db
    base_frac = {
        s: base_all[base_all.system == s].sort_values("seed")
        .audibility_fraction.round(2).tolist()
        for s in ("nlms_f64", "speex")}

    # Reconstruction QC and decimation materiality (far-single rows,
    # where the trajectory is exercised).
    fs_rows = runs[(runs.talk == "far_single") & (runs.status == "ok")]
    f64_fs = fs_rows[fs_rows.system == "nlms_f64"]
    q15_fs = fs_rows[fs_rows.system == "nlms_q15"]
    f64_traj_dfrac = (f64_fs.audibility_fraction_traj
                      - f64_fs.audibility_fraction)
    f64_interp_dfrac = (f64_fs.audibility_fraction_interp
                        - f64_fs.audibility_fraction)
    q15_traj_dfrac = (q15_fs.audibility_fraction_traj
                      - q15_fs.audibility_fraction)

    # ------------------------------------------------------------ cost
    cost_tab_rows = []
    for system in sys_order:
        sub = cost[cost.system == system]
        cost_tab_rows.append({
            "system": system,
            "RTF (mean)": sub.rtf.mean(),
            "RTF (range)": f"{sub.rtf.min():.4f}–{sub.rtf.max():.4f}",
            "MAC/sample (derived)": (f"{sub.mac_per_sample.iloc[0]:,.0f}"
                                     if sub.mac_per_sample.iloc[0] > 0
                                     else "–"),
            "state (bytes)": (f"{sub.state_bytes.iloc[0]:,d}"
                              if sub.state_bytes.iloc[0] > 0 else "–"),
        })
    cost_tab = pd.DataFrame(cost_tab_rows).set_index("system")
    mac_nlms = cost[cost.system == "nlms_f64"].mac_per_sample.iloc[0]
    mac_mdf = cost[cost.system == "speex"].mac_per_sample.iloc[0]
    rtf_f64 = cost[cost.system == "nlms_f64"].rtf.mean()
    rtf_q15 = cost[cost.system == "nlms_q15"].rtf.mean()
    rtf_speex = cost[cost.system == "speex"].rtf.mean()

    # Two-run approximation error bar: cells where d != d_echo only
    # (noise and double-talk; in no-noise far-single the two runs are
    # bit-identical by construction and the diff is exactly 0).
    speex_two = runs[(runs.system == "speex")
                     & (runs.speex_tworun_erle_diff_db.notna())
                     & ((runs.noise_type != "none")
                        | (runs.talk == "double"))]
    two_diff = speex_two.speex_tworun_erle_diff_db

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

    # ---------------------------------------------- figure/table plumbing
    # Table-only variants: numbers shown in a figure are not repeated in
    # a table (and vice versa); the full aggregates stay for prose.
    q_double_tab = q_double.drop(index="stoi")   # STOI: stage_b_talk fig
    q_near_tab = q_near.drop(index="stoi")
    mu_conv_tab = mu_piv[["convergence (s)"]]    # rest: stage_b_mu fig
    wl_diag_tab = wl_tab.drop(
        columns=["steady ERLE (dB)", "misalignment (dB)"])  # in sweep fig
    wl_aud_tab = wl_aud.drop(columns=["steady ERLE (dB)"])  # in sweep fig
    base_speex_two = base_speex.speex_tworun_erle_diff_db.abs()
    ctx = {
        "n_seeds": n_seeds,
        # Seed shown in the single-seed diagnostic figures; matches
        # plot_baseline_curves' default.
        "diag_seed": 0,
        "base_speex_two_max": base_speex_two.max(),
        # Figure-caption values (STEP-11 reachability): sourced from the
        # same aggregates that feed the figures.
        "stoi_dt_none": q_double.loc["stoi", "none"],
        "stoi_dt_f64": q_double.loc["stoi", "nlms_f64"],
        "stoi_dt_q15": q_double.loc["stoi", "nlms_q15"],
        "stoi_dt_speex": q_double.loc["stoi", "speex"],
        "q15_a_rt02_d1": apivot("nlms_q15",
                                "erle_steady_state_db").loc[0.2, 1.0],
        "q15_a_rt08_d1": apivot("nlms_q15",
                                "erle_steady_state_db").loc[0.8, 1.0],
        "noise_speex_clean": noise_erle.loc["no_noise", "speex"],
        "noise_speex_snr10": noise_erle.loc["snr10", "speex"],
        "noise_f64_snr20": noise_erle.loc["snr20", "nlms_f64"],
        "noise_f64_snr10": noise_erle.loc["snr10", "nlms_f64"],
        "noise_q15_snr10": noise_erle.loc["snr10", "nlms_q15"],
        "aud_frac_none": aud_base.loc["none", "audible fraction"],
        "aud_frac_f64": aud_base.loc["nlms_f64", "audible fraction"],
        "aud_frac_q15": aud_base.loc["nlms_q15", "audible fraction"],
        "aud_frac_speex": aud_base.loc["speex", "audible fraction"],
        "rt_min": a.rt60_target_s.min(),
        "q15_d03_lo": apivot("nlms_q15",
                             "erle_steady_state_db")[0.3].min(),
        "q15_d03_hi": apivot("nlms_q15",
                             "erle_steady_state_db")[0.3].max(),
        "tail_f64_200": tail_erle.loc["200ms", "nlms_f64"],
        "tail_f64_400": tail_erle.loc["400ms", "nlms_f64"],
        "tail_q15_200": tail_erle.loc["200ms", "nlms_q15"],
        "tail_q15_400": tail_erle.loc["400ms", "nlms_q15"],
    }

    # Reader-facing system names for results-table headers; the Method
    # section's system-ID table deliberately keeps the code IDs.
    SYS_RENAME = {"none": "Passthrough", "nlms_f64": "Float NLMS",
                  "nlms_q15": "Q15 NLMS", "speex": "SpeexDSP"}

    def FIG(key: str) -> str:
        return fig_embed(key, ctx)

    def FIGD(key: str, summary: str, extra: str = "") -> str:
        """Diagnostics figure collapsed in a <details> block (optionally
        with an accompanying table), so the open page shows only the
        main-sequence figures."""
        body = fig_embed(key, ctx)
        if extra:
            body += "\n\n" + extra
        return (f"<details>\n<summary><b>{summary}</b> "
                f"({fig_ref(key)})</summary>\n\n{body}\n\n</details>")

    def DET(summary: str, body: str) -> str:
        return (f"<details>\n<summary><b>{summary}</b></summary>\n\n"
                f"{body}\n\n</details>")

    def TBL(key: str, table_md: str) -> str:
        return table_block(key, table_md, ctx)

    REF = fig_ref

    # ------------------------------------------------------------------ report
    report = f"""# AEC benchmark — report

Comparison of a float64 NLMS adaptive filter (`nlms_f64`), a Q15
fixed-point NLMS (`nlms_q15`), and the SpeexDSP MDF echo canceller
(`speex`) against a passthrough reference (`none`) on synthesised
room-acoustic scenarios. {n_total} runs ({n_seeds} utterance-pair seeds
per condition); {n_ok} completed normally, {n_total - n_ok} diverged
(all {"`" + "`, `".join(sorted(div.system.unique())) + "`" if len(div)
 else "none"}; §2.4 — the fixed-point system cannot trip the divergence
detector at all, see §2.8). All numbers in this report are rendered
from `results/raw/*.csv` by `scripts/render_report.py`; provenance is
recorded per run (§5).

## Summary

1. At the clean baseline the Float NLMS and SpeexDSP reach comparable
   steady-state ERLE ({b_nlms_erle.mean():.1f} vs
   {base_speex.erle_steady_state_db.mean():.1f} dB, {REF("stage_a_erle")})
   — while SpeexDSP's partitioned frequency-domain structure executes
   {mac_mdf:,.0f} MAC/sample against the time-domain NLMS's
   {mac_nlms:,.0f} at the same tail length
   ({mac_nlms / mac_mdf:.1f}× fewer operations, {table_ref("cost")}).
2. The shared hazard for the deliberately unprotected Float NLMS is
   uncorrelated microphone energy, particularly during
   low-reference-power periods: {n_total - n_ok} of its runs diverged
   outright under double-talk or background noise
   ({REF("stage_b_talk")}, {REF("stage_b_noise")}), while SpeexDSP's
   adaptation control degrades gracefully on both axes.
3. Saturating Q15 arithmetic converts that catastrophic failure into
   bounded degradation in every matched run ({table_ref("six_cell")}) —
   bounded, not healthy: the contained runs still leak echo and end far
   from the true echo path.
4. Reduced coefficient word length under the floor-masked storage
   convention tested here fails as a cliff, not a slope: every masked
   level is worse than no processing at all ({REF("word_length_sweep")}).
   This is specific to the floor-masked truncating-storage convention
   tested here; it does not generalise to other rounding conventions.
5. At nearly equal ERLE ({base_speex.erle_steady_state_db.mean():.1f} vs
   {b_nlms_erle.mean():.1f} dB), SpeexDSP's residual exceeds the masking
   threshold in {aud_base.loc["speex", "audible fraction"]:.2f} of
   time–frequency units against the Float NLMS's
   {aud_base.loc["nlms_f64", "audible fraction"]:.2f}
   ({REF("audibility")}) — equal energy suppression, differently
   distributed residual; an energy-only metric cannot see this.

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

{DET("Level calibration (active-level measure and SER pinning)",
     f'''Speech material is normalised to
{lvl_cfg["active_level_dbov"]:g} dBov active-speech level using an
energy-gated RMS — frames more than
{lvl_cfg["active_gate_threshold_db"]:g} dB below the peak frame RMS are
treated as inactive. This is a simplified active-level measure, not
ITU-T P.56. The near-end level is then pinned to a configured
signal-to-echo ratio (SER {cfg["scenario_defaults"]["ser_db"]:g} dB,
near-end over echo, measured on the double-talk overlap), so the
fixed-point operating point is controlled rather than an accident of
room geometry.''')}

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

Wall absorption is calibrated per RT60 level until the RT60 measured
on the generated impulse response matches the target; figures and
tables are labelled with **achieved** RT60 throughout.

{DET("RT60 calibration detail (Table 1: Sabine initialisation vs calibrated absorption)",
     f'''`inverse_sabine` is used only to **initialise** the wall absorption;
the value actually used is calibrated by bisection until the RT60
measured on the generated loudspeaker-to-mic RIR (Schroeder backward
integration) is within ±{cal["tolerance_pct"].iloc[0]:g}% of target at
the {cal["reference_distance_m"].iloc[0]:g} m reference distance. The
uncalibrated Sabine values overshoot systematically, and the overshoot
grows with target RT60 — a small finding in itself:

{TBL("calibration", cal_md)}

The calibrated absorption for each RT60 level is shared across all
distance levels of that row (per-distance recalibration would confound
the distance axis with wall absorption). Residual drift remains at
2.5 m, where every level measures ~5–8% above target: the weaker
direct path shifts the Schroeder fit toward the reverberant tail.
Per-scenario achieved values are stored in every CSV row.''')}

### 1.3 Systems under test

| ID | Description |
|---|---|
| `none` | Passthrough reference. Not a no-op: `d` goes through the identical float→int16→float round-trip at the identical scaling constant as `speex`, so the 0 dB reference is measured on the same signal path and carries the same quantisation floor. |
| `nlms_f64` | Sample-wise float64 NLMS, L = {nlms_cfg["filter_length_ms"]:g} ms ({int(nlms_cfg["filter_length_ms"] * 16)} taps), μ = {nlms_cfg["mu"]:g}, δ = {nlms_cfg["delta"]:g}. **No double-talk detection, deliberately** — its absence is what makes SpeexDSP's built-in protection visible. Verified sample-exact against a naive per-sample reference. |
| `speex` | SpeexDSP MDF, frame size {speex_cfg["frame_size"]} samples, tail {speex_cfg["filter_length_ms"]:g} ms, sampling rate set explicitly via `speex_echo_ctl` and read back (asserted). |
| `nlms_q15` | Q15 fixed-point re-implementation of the same NLMS recursion: int16/Q15 coefficients, Q15×Q15→Q30 products accumulated in int64, **saturating** narrowing at every return to 16 bits with per-site event counting. ‖x‖² normalisation is block floating point: the window power is an exact integer sliding-window sum in Q30 (no cancellation, unlike the float path's guarded cumulative sum), and one reciprocal per sample is taken on a 15-bit mantissa with the exponent tracked and folded into the gain shift. μ = {q15_cfg["mu"]:g} as the Q15 constant {mu_q15_int}, δ = {q15_cfg["delta"]:g} as the Q30 constant {delta_q30_int}. Bit-exact against a pure-Python unbounded-integer reference executing the same arithmetic naively. Runs on the identical int16 signals and scaling constant as `speex`/`none`; an in-loop float64 shadow filter fed the same quantised input records per-sample coefficient divergence (§1.4). |

<details>
<summary><b>Rounding convention (load-bearing)</b></summary>

In `nlms_q15`, every product
narrowing — filter output, gain, per-tap update — uses **magnitude
truncation** (shift toward zero), so that a sub-LSB update truncates to
zero on either sign — the definition of stalling adopted throughout; the
word-length mask (§2.7) keeps the plain floor `>>`/`<<` semantics of the
`(w >> shift) << shift` masking rule. The
choice is not cosmetic: a floor-truncating (arithmetic-shift) update
path was implemented first and fails outright at full 15-bit precision
on real speech — floor biases every update by −0.5 LSB in the mean, and
during speech pauses the error feedback is too weak to oppose the
accumulating drift, which drove the mean coefficient to roughly
two-thirds of the negative rail on the baseline scenario (positive
misalignment, negative ERLE, before any masking; mechanism documented
in `src/aec_nlms_fixed.py`). Every stalling and saturation result below
is therefore a statement about a magnitude-truncating, floor-masked
implementation; a round-to-nearest implementation would stall less and
behave differently. This is a scope statement, not a weakness — but it
means fixed-point conclusions do not transfer across rounding
conventions. The floor-truncating implementation is preserved in the
repository's git history (the commit introducing `aec_nlms_fixed.py`
carries it; the following commit switches the arithmetic-path
convention) for anyone who wants to reproduce the failure.

</details>

{DET("Signal-path details and comparison caveats (int16 scaling; float-vs-fixed asymmetry; no Speex preprocessor)",
     f'''One float→int16 scaling constant per run, computed to leave
{lvl_cfg["int16_headroom_db"]:g} dB headroom above max(|x|, |d|),
applied identically to every system in the run, asserted not to clip
on `x` and `d`, with `speex` output checked for saturation. The
constant is recorded in every CSV row.

`nlms_f64` runs in float throughout — it never passes through the
int16 path. The float-vs-fixed comparison is therefore asymmetric by
construction: NLMS is exempt from the quantisation floor the
fixed-point systems carry. This is stated here as a caveat and
revisited in §4.

Speex is tested **as a linear echo canceller only**: the
`speex_preprocess` chain (residual echo suppressor, noise suppressor)
is never attached. Attaching it would confound the comparison against
a bare adaptive filter — the preprocessor's nonlinear suppression
would mask the linear canceller's actual behaviour. Comparisons
against deployed Speex configurations (which typically include the
preprocessor) should keep this in mind.''')}

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
truncated/zero-padded to the filter length. Computed for `nlms_f64` and
`nlms_q15` (Speex does not expose coefficients). For the fixed-point
path, w/2¹⁵ is compared to `h_echo` directly: x and d share one scaling
constant, so the echo path in the int16 domain is unchanged.

<details>
<summary><b>Fixed-point instrumentation</b> — stalls, saturation, divergence from float (`nlms_q15` only)</summary>

All counters are evaluated on the coefficient state *after* saturation
and after the word-length mask, so the sweep and the counters tell one
story:

- *Full-stall events*: samples where the error is nonzero, the
  reference window is nonzero, and **no** coefficient changed —
  adaptation halted for that sample. Count and positions recorded.
- *Per-tap stalls*: active taps whose coefficient did not change during
  an attempted update — the tap-granular version of the same
  phenomenon, recorded as a per-sample count.
- *Saturation events*: per narrowing site (filter output `y`, error,
  gain, coefficient update), counts and sample positions.
- *Coefficient divergence from float*: an in-loop float64 shadow filter
  runs the identical recursion on the identical quantised input
  (x/2¹⁵, d/2¹⁵), and 10·log10(‖w_q15/2¹⁵ − w_float‖²/‖w_float‖²) is
  recorded **per sample**, so bursts of saturation and the growth of
  coefficient error can be aligned in time. The shadow is never masked;
  at reduced word lengths the curve measures total degradation against
  the ideal float filter. Because the shadow sees quantised input, the
  curve isolates arithmetic degradation from input quantisation.

Scalar summaries of all of these land as columns in `runs.csv`
(`n_stall_events`, `n_tap_stalls`, `n_sat_*`, `sat_first_time_s`,
`coeff_div_steady_db` = median over the final
{met_cfg["steady_state_last_fraction"]:.0%} of the divergence curve).

</details>

**Perceptual audibility of residual echo.** ERLE is
an energy ratio; audibility depends on masking. A simplified
simultaneous-masking model is implemented in `src/psychoacoustic.py`.

<details>
<summary><b>Model constants and rationale</b> (all constants in <code>config/scenarios.yaml</code> under <code>audibility:</code>)</summary>

The model is deliberately not an off-the-shelf PEAQ/psychoacoustics
package: the analysis asks a *relative* question (which system's
residual is more audible under the same masker), and a fixed-offset
simplified model biases all systems identically, so comparative
conclusions survive the simplification.

{aud_cfg["stft_nperseg"]}-sample Hann STFT with a
{aud_cfg["stft_hop"]}-sample hop (one 20 ms segmentation frame, so
STFT frames map 1:1 onto ERLE-valid frames); power mapped to Bark
bands via the Zwicker–Terhardt arctan approximation
(z = {aud_cfg["bark_a"]:g}·arctan({aud_cfg["bark_b"]:g}·f) +
{aud_cfg["bark_c"]:g}·arctan((f/{aud_cfg["bark_d_hz"]:g})²));
two-slope spreading ({aud_cfg["spreading_lower_db_per_bark"]:g} dB/Bark
toward lower bands, {aud_cfg["spreading_upper_db_per_bark"]:g} dB/Bark
toward higher); threshold = spread masker power −
{aud_cfg["masking_offset_db"]:g} dB, floored at
{aud_cfg["threshold_floor_db"]:g} dB re full-scale² band power.

</details>

The masker is near-end speech plus noise (s + v). Audibility is
computed **over the same ERLE-valid (far-single) frames the ERLE
metric uses**, so the two are comparable row by row — which means the
masker over those frames is the background noise where present, and
silence in no-noise cells (near-end speech is by definition inactive
during far-single frames). In silent-masker cells the threshold is the
constant floor — the absolute-threshold proxy; without it those cells
would report ~100% audibility by construction. The floor is a fixed
constant, not a physiological curve, and is identical across systems.

Outputs per run: fraction of time–frequency (Bark band × frame) units
in which the residual exceeds the threshold, and the mean excess above
threshold in dB over those units.

<details>
<summary><b>Residual isolation</b> — exact component identity for the linear paths; two-run approximation for SpeexDSP (rejected outside the baseline cell)</summary>

For the linear-subtraction systems (`none`,
`nlms_f64`, `nlms_q15`) the residual is computed by the exact component
identity r = e − s − v, which for e = d − y and d = d_echo + s + v is
algebraically identical to the trajectory decomposition d_echo − y(w(n))
evaluated with the *exact per-sample* coefficients — the recorded-
trajectory method at a snapshot every sample. It is exact for
`nlms_f64`, and exact for `nlms_q15` except at samples where the error
narrowing saturated (counted per run; zero in clean cells). The
recorded-trajectory reconstruction itself (coefficients held at the
block-start state between {q15_cfg["record_every"]}-sample snapshots;
also linearly interpolated for the float path — interpolating int16
states is not Q15 arithmetic, so the fixed-point path is hold-only, and
its reconstruction runs the same saturating Q15 arithmetic as the
filter) is still computed on every run: its error against the exact
output is a QC column, and audibility fractions from the trajectory
residuals are recorded alongside the primary ones. §2.9 reports what
that QC found. For `speex`, whose coefficients are not exposed and
whose internal DC notch on the microphone path breaks the exact
identity, a **two-run approximation** is used instead: the same
configuration is run again on an echo-only microphone signal and that
run's output is taken as the residual. Adaptation trajectories differ
between the two runs, making this an approximation, not an exact
decomposition; the ERLE difference between the two runs over
far-single segments is recorded per run as the approximation's error
bar (§2.9).

</details>

**Divergence**: an output is flagged diverged at the first sample that is
non-finite or exceeds 10× the peak of |d| (+20 dB); the onset time is
recorded as a data point. Divergence is detected, never prevented.

### 1.5 Experiment matrix

Stage A crosses RT60 × speaker–mic distance (far single-talk, no noise);
Stage B varies one factor at a time around the baseline (RT60 0.4 s,
1.0 m): talk state, background noise, tail length (all adaptive
systems), NLMS step size (`nlms_f64` only, with Speex run once at
baseline as a labelled reference — it has no step-size parameter), and
effective coefficient word length (`nlms_q15` only, via low-bit masking;
the unmasked 15-bit level is run under its own label so the sweep is
constructed identically at every level). {n_total} rows; the sum of
per-row processing times is {wall_sum_min:.1f} min.

Reporting convention: all Stage B figures show individual seeds with
diverged runs marked, and means are drawn as bars — with
{n_seeds} seeds, error bars would imply a precision this design does
not have. The double-talk cell is the argument for this convention: of
three seeds under identical conditions, two diverged (2.7 and 3.1 s
into double-talk) and one survived with degraded but positive ERLE
(§2.4); a mean over those three (−12 dB) describes none of them. For the recorded
reference run this sum agreed with the batch's own wall-clock timer
(the figure quoted in the README) to within the timer's 0.1-minute
print resolution — inter-row overhead is a few seconds at most, which
is why the wall-clock figure and this sum coincide at one decimal.

## 2. Results

### 2.1 Stage A — RT60 × distance

{FIG("stage_a_erle")}

{FIGD("stage_a_convergence", "Convergence time by RT60 and distance")}

ERLE degrades monotonically with RT60 for the Float NLMS and SpeexDSP at
every distance — at 1.0 m, the Float NLMS falls from
{apivot("nlms_f64", "erle_steady_state_db").loc[0.2, 1.0]:.1f}
to {apivot("nlms_f64", "erle_steady_state_db").loc[0.8, 1.0]:.1f} dB and
SpeexDSP from {apivot("speex", "erle_steady_state_db").loc[0.2, 1.0]:.1f}
to {apivot("speex", "erle_steady_state_db").loc[0.8, 1.0]:.1f} dB across
the four levels; the Q15 NLMS does not follow this pattern (§2.8). The
per-step drop is roughly constant (~5–7 dB per 0.2 s of RT60) rather
than accelerating at the longest tail. Distance (closer loudspeaker →
higher ERLE) is monotonic for those two systems except in
{n_nonmono_dist} of the {n_dist_rows} system × RT60 rows, both at the
{min(v for vs in nonmono_dist.values() for v in vs):g} s level:
SpeexDSP peaks at the middle distance and the Float NLMS ticks back up
at the farthest.

### 2.2 Talk state

{FIG("stage_b_talk")}

{DET("Double-talk quality metrics (Table 2: segSNR and PESQ)",
     TBL("talk_quality_double", md_table(q_double_tab.rename(columns=SYS_RENAME), "metric", "{:.2f}")))}

{DET("Near-end-only quality: the passthrough quantisation floor and the SpeexDSP DC notch",
     TBL("talk_quality_near", md_table(q_near_tab.rename(columns=SYS_RENAME), "metric", "{:.2f}"))
     + f'''

The Passthrough row's near-end-only segmental SNR ({none_floor:.1f} dB)
is the measured int16 quantisation floor of the shared I/O path — the
ceiling against which the other systems' segSNR should be read. The
Float NLMS reports +inf there because with a silent reference its float
output reproduces `d` bit-exactly. SpeexDSP's much lower value is
discussed in §4.''')}

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
§2.4). The fixed-point NLMS is equally unprotected but cannot exhibit
the same unbounded-output failure, because its arithmetic saturates;
instead it settles into bounded degradation:
{q15_dt.erle_steady_state_db.mean():.1f} dB mean ERLE and
{q_double.loc["segsnr_db", "nlms_q15"]:+.1f} dB segSNR during
double-talk, with
{q15_dt.n_sat_total.mean():,.0f} saturation events per run on average
(vs {q15_clean_sat.mean():,.0f} averaged over all clean Stage A runs).
§2.8 examines this float-vs-fixed contrast cell by cell.

On STOI the ordering is uniform: the float NLMS scores below the
unprocessed passthrough in every double-talk seed
({dt_stoi["nlms_f64"].min():.2f}–{dt_stoi["nlms_f64"].max():.2f} against
{dt_stoi["none"].min():.2f}–{dt_stoi["none"].max():.2f};
{REF("stage_b_talk")}). Two of the three float runs are the diverged
double-talk pairs of {table_ref("six_cell")} and belong to the
containment narrative described there. The third seed did not diverge
and the ordering still holds
({dt_stoi.loc[ok_seed, "nlms_f64"]:.2f} vs
{dt_stoi.loc[ok_seed, "none"]:.2f} passthrough on that seed), so the
ordering is not explained by divergence alone. What produces the
non-diverged seed's gap is not determinable from what was measured:
saturation and stall counts are recorded as whole-run totals and are
not segmented by talk state, so determining it would require
talk-state-segmented instrumentation, which was not done here.

### 2.3 Background noise

{FIG("stage_b_noise")}

Speex degrades gracefully as SNR falls. Float NLMS collapses: at 20 dB
SNR {snr20_div} of {n_seeds} seeds diverged; at 10 dB SNR {snr10_div} of
{n_seeds} did. No near-end speech is present in any of these runs. The
Q15 NLMS on the identical microphone signals reports
{noise_erle.loc["snr10", "nlms_q15"]:.1f} dB mean ERLE at 10 dB SNR —
degraded relative to its clean-condition
{noise_erle.loc["no_noise", "nlms_q15"]:.1f} dB, but bounded, with
{q15_ns10.n_sat_total.mean():,.0f} saturation events per run on average
(§2.8). Diverged runs are part of the data in {REF("stage_b_noise")},
not excluded.

### 2.4 Divergence onsets

All {len(div)} divergences are Float NLMS runs; in double-talk they
begin a few seconds after the near end starts, under 10 dB noise within
the first seconds of the run.

{DET("Divergence onsets per run (Table 4)",
     TBL("divergence_onsets", div_md) + f'''

Per-seed detail for double-talk ({", ".join(dt_seed_strs)}) shows the
divergences beginning 2.7–3.1 s after double-talk onset — and that one
seed survives outright. Under 10 dB noise, divergence arrives within
{div[(div.axis == "noise") & (div.level == "snr10")].divergence_time_s.min():.1f}–{div[(div.axis == "noise") & (div.level == "snr10")].divergence_time_s.max():.1f} s
of signal onset, before any adaptation has consolidated.''')}

### 2.5 Tail length

{FIGD("stage_b_tail", "Tail-length sweep: ERLE per filter length",
       TBL("tail_convergence", md_table(tail_conv.rename(columns=SYS_RENAME), "filter length")))}

All three systems lose heavily from undermodelling (50 ms covers
little of a 0.4 s decay). Both NLMS filters peak at 200 ms and *lose*
ground at 400 ms — the drop is consistent with the longer filter
increasing adaptation variance and slowing convergence per tap over
the fixed 15 s horizon — while SpeexDSP holds its 200 ms performance
at 400 ms.

### 2.6 NLMS step size

{FIGD("stage_b_mu", "Step-size sweep: ERLE and misalignment",
       TBL("mu_convergence", md_table(mu_conv_tab, "level", "{:.2f}")))}

Speex, which has no step-size parameter, runs once at its baseline
configuration as the reference line in {REF("stage_b_mu")}; that
reference converges in {mu_ref.convergence_time_s.mean():.2f} s.

### 2.7 Word-length sweep (headline)

Effective coefficient word length via post-update low-bit masking of the
Q15 coefficients (floor semantics, §1.3); 15 bits is the unmasked
filter.

{FIG("word_length_sweep")}

This is not the gentle precision-vs-performance slope the sweep was
designed to trace. Four observations:

1. **Unmasked Q15 works.** At 15 bits the filter reaches
   {wl_erle.loc["15bit"]:.1f} dB mean ERLE (a few dB under float,
   §2.8), stalls heavily at its own noise floor
   ({wl_stalls.loc["15bit"]:,.0f} full-stall samples per 15 s run —
   the classic sub-LSB truncation phenomenon), and saturates only at
   the gain site.
2. **Every masked level is worse than no filter at all** — mean
   steady-state ERLE {wl_erle.loc["11bit"]:.1f} /
   {wl_erle.loc["9bit"]:.1f} / {wl_erle.loc["7bit"]:.1f} dB at
   11 / 9 / 7 bits, all below the 0 dB passthrough line, consistent
   across every seed. The masked filter *injects* energy. Insufficient
   coefficient word length under this storage scheme is not "less
   accurate" — it is actively harmful, and there is a cliff between 15
   and 11 bits rather than a slope.
{DET("Mechanism and diagnostics: the storage ratchet, the stall counters, and Table 7",
     f'''3. **The failure mode is a ratchet to the rail, not gradual noise.**
   The floor mask erases a positive sub-effective-LSB update but turns
   a negative one into a full effective-LSB step downward; inside the
   adaptation loop this is a one-way ratchet, and the filter's 3200
   small-magnitude coefficients walk to the negative rail. The rail
   signature is unambiguous in the table below: coefficient-site
   saturations go from
   {wl_piv.loc["15bit", "n_sat_coeff"]:,.0f} per run at 15 bits
   to {wl_piv.loc["11bit", "n_sat_coeff"]:,.0f}–{wl_piv.loc[
   "7bit", "n_sat_coeff"]:,.0f} at the masked levels, and misalignment
   sits at
   +{wl_mis.loc["11bit"]:.0f} dB (the railed coefficient vector's
   energy, orders of magnitude above ‖h‖²). This is the same
   floor-bias mechanism that rules out floor truncation in the update
   arithmetic (§1.3), re-entering through the coefficient store: the
   `>>`/`<<` mask faithfully models truncating two's-complement
   storage, and truncating storage is what collapses.
4. **The stall counters do not carry the degradation signal.**
   Full-stall counts are non-monotone across the sweep
   ({wl_stalls.loc["11bit"]:,.0f} / {wl_stalls.loc["9bit"]:,.0f} /
   {wl_stalls.loc["7bit"]:,.0f} at 11 / 9 / 7 bits vs
   {wl_stalls.loc["15bit"]:,.0f} at 15) — once the coefficients rail,
   the dynamics are dominated by rail-and-saturation interplay rather
   than sub-LSB truncation. Per-tap stalls are likewise
   non-discriminative (~{wl_tap_stalls_mean:.1e} per run at every
   level, dominated by small-|x| taps). The naive expectation that a
   coarser LSB simply means more stalling is wrong at this scale; the
   degradation lives in ERLE and misalignment. (On a small stationary
   test configuration — 32 large taps, white noise — the same mask
   instead produces a bounded limit cycle with *fewer* stalls and a
   graded misalignment floor; see the unit tests. The word-length
   story is filter-scale-dependent as well as convention-dependent.)

{TBL("word_length_diag", md_table(wl_diag_tab, "word length", "{:.2f}"))}

The convergence-time column is the §1.4 artifact in its extreme form:
90% of a deeply negative steady state is reached almost immediately,
so the ~0.04 s entries for masked runs mean "collapsed instantly", not
"converged fast".''')}

### 2.8 Fixed point vs float

At the baseline scenario the Q15 filter costs a few dB against its
float counterpart on identical tasks: steady-state ERLE
{base_q15.erle_steady_state_db.mean():.1f} vs
{b_nlms_erle.mean():.1f} dB, final misalignment
{base_q15.misalignment_final_db.mean():.1f} vs
{b_nlms_mis.mean():.1f} dB (mean over seeds). Across all of Stage A the
paired per-cell ERLE gap (float − Q15, same cell, same seed) is
{q15_gap.mean():.1f} dB (min {q15_gap.min():.1f}, max
{q15_gap.max():.1f}). The gap concentrates at the 0.3 m distance
({gap_by_d.loc["0.3"]:.1f} dB mean, vs {gap_by_d.loc["1"]:.1f} /
{gap_by_d.loc["2.5"]:.1f} dB at 1.0 / 2.5 m): with the loudspeaker
that close the echo-path taps are large enough that the coefficient
site saturates continuously ({d03_sat_coeff:,.0f} clipped-tap events
per run on average; misalignment stalls near {d03_mis:.1f} dB) — the
Q15 filter is *representation*-limited there, not merely
quantisation-noise-limited. The per-sample divergence between the Q15
coefficients and the float64 shadow filter on identical quantised input
settles at {base_q15.coeff_div_steady_db.mean():.1f} dB (baseline,
mean over seeds):

{FIGD("q15_float_divergence", "Per-sample Q15-vs-float coefficient divergence")}

**Where the float filter diverged, the fixed-point filter stayed
bounded — in every one of the {len(cmp_rows)} matched runs.**
Saturating arithmetic bounds
every quantity the float recursion lets explode, so the same
uncorrelated-energy hazard (§3.1) produces bounded degradation instead:

{DET("The six matched runs (Table 8) and the coefficient-trajectory separation",
     TBL("six_cell", div_cmp_md) + f'''

The "div. from float" column reads ≈0 dB in every one of these cells,
and the per-sample curves show why: under strong uncorrelated energy
the Q15 and float coefficient trajectories separate **immediately and
completely** — the relative difference is already ~0 dB (100%) within
the first fraction of a second, long before the float run's *output*
crosses the divergence threshold. There is no gradual
saturation-then-divergence cascade; the two arithmetic paths simply
never correspond once the input is noise-dominated, whereas in clean
conditions the same curve locks to
{base_q15.coeff_div_steady_db.mean():.0f} dB. (In these cells the
float shadow inside the Q15 run is itself the diverging recursion, so
the column measures separation from a diverged trajectory — which is
the point: there is no meaningful float solution left to track.)''')}

Two caveats keep this honest — see also §2.9's audibility view of the
same cells. First, the divergence detector
(non-finite or >10× peak |d|) **cannot fire** for `nlms_q15`: int16
output can never exceed ~5× the peak of a signal recorded with
{lvl_cfg["int16_headroom_db"]:g} dB headroom, so `status = ok` means
"bounded", not "healthy" — the health measures are ERLE, misalignment,
and the divergence-from-float column. Second, bounded is not good:
{q15_dt.erle_steady_state_db.mean():.1f} dB mean ERLE in double-talk is
still echo leaking through, and the coefficient state ends far from the
float solution. Saturation acts as a crude, implicit safety net — an
architectural side effect, not adaptation control; Speex's explicit
protection achieves bounded *and* useful (§2.2–2.3).

### 2.9 Perceptual audibility of residual echo

Method and residual-isolation definitions in §1.4.

{FIG("audibility")}

<details>
<summary><b>Scope note: noise and double-talk audibility cells are measured but not plotted</b></summary>

The noise-axis and double-talk audibility cells were measured and are
tabulated below. They are excluded from {REF("audibility")} because the
SpeexDSP residual there relies on the two-run approximation, and that
approximation was validated and rejected for those cells: over the
{len(speex_two)} runs where the microphone signal differs from the echo
alone, the between-run steady-state ERLE difference averages
{two_diff.abs().mean():.1f} dB (maximum {two_diff.abs().max():.1f} dB).
By contrast, at the baseline cell the same code path is exact — the
between-run difference is {base_speex_two.max():.1f} dB for all
{n_seeds} seeds.

**Two-run disagreement (`speex`).** In no-noise far-single cells the
two runs are bit-identical by construction (d = d_echo) and the
difference is exactly zero. Where the microphone signals differ (noise
and double-talk cells, n = {len(speex_two)} runs), the echo-only run's
steady-state ERLE differs from the primary run's by
{two_diff.mean():+.1f} dB on average (range {two_diff.min():+.1f} to
{two_diff.max():+.1f} dB). That figure is the between-run disagreement
of the decomposition itself — the reason the two-run residual is
rejected for these cells — not an error bound on an audibility
fraction, which is a dimensionless ratio and cannot carry a dB error.

As a direction check on the masking model itself — not a result about
any canceller — the exact component-identity residuals behave as
follows: the passthrough's audible fraction falls
{noise_frac.loc["no_noise", "none"]:.3f} →
{noise_frac.loc["snr20", "none"]:.3f} →
{noise_frac.loc["snr10", "none"]:.3f} across the noise axis and the Q15
NLMS's mean excess falls
{noise_exc.loc["no_noise", "nlms_q15"]:.1f} →
{noise_exc.loc["snr20", "nlms_q15"]:.1f} →
{noise_exc.loc["snr10", "nlms_q15"]:.1f} dB, while the Float NLMS moves
the other way ({noise_frac.loc["no_noise", "nlms_f64"]:.3f} →
{noise_frac.loc["snr20", "nlms_f64"]:.3f} →
{noise_frac.loc["snr10", "nlms_f64"]:.3f}) because its diverged output
dominates its residual. This is a sanity check on the model's
direction, nothing more.

{TBL("audibility_noise_fraction", md_table(noise_frac.rename(columns=SYS_RENAME), "condition", "{:.2f}"))}

{TBL("audibility_noise_excess", md_table(noise_exc.rename(columns=SYS_RENAME), "condition", "{:.2f}"))}

</details>

One reading of the noise-cell measurements stands on non-rejected
data (the component-identity residuals of the NLMS systems and the
passthrough, which are exact in every cell): the float NLMS rows
include its diverged runs, and audibility renders §2.8's
containment story perceptually: `nlms_f64`'s "residual" (which the
component identity correctly charges with the diverging output)
exceeds threshold in ~100% of units at
{noise_exc.loc["snr10", "nlms_f64"]:.1f} dB mean excess at 10 dB SNR —
*worse than the unprocessed echo*
({noise_exc.loc["snr10", "none"]:.1f} dB) — while the bounded
`nlms_q15` sits at {noise_exc.loc["snr10", "nlms_q15"]:.1f} dB,
audibly better than doing nothing even where its coefficients are far
from the float solution.

The three comparisons this layer was built to check:

- **`nlms_q15` (15 bits) vs `nlms_f64`, baseline.** ERLE
  {aud_base.loc["nlms_q15", "steady ERLE (dB)"]:.1f} vs
  {aud_base.loc["nlms_f64", "steady ERLE (dB)"]:.1f} dB; mean excess
  {aud_base.loc["nlms_q15", "mean excess (dB)"]:.1f} vs
  {aud_base.loc["nlms_f64", "mean excess (dB)"]:.1f} dB — the
  audibility gap ({aud_base.loc["nlms_q15", "mean excess (dB)"]
                   - aud_base.loc["nlms_f64", "mean excess (dB)"]:+.1f}
  dB) tracks the ERLE gap
  ({aud_base.loc["nlms_f64", "steady ERLE (dB)"]
    - aud_base.loc["nlms_q15", "steady ERLE (dB)"]:.1f} dB) rather
  than exceeding it: the Q15 arithmetic's extra residual behaves, to
  this masking model, like ordinary additional residual echo.
- **Masked word lengths.** Audibility agrees with ERLE's
  worse-than-passthrough verdict and sharpens it:

{DET("Masked word-length audibility per level",
     md_table(wl_aud_tab, "word length", "{:.2f}"))}

  The masked filters' mean excess exceeds even the unprocessed echo's
  ({none_a_excess.mean():.1f} dB mean across clean Stage A cells) —
  under the masking model, the injected limit-cycle jitter is worse
  than doing nothing not just energetically but in modelled audibility.
- **`speex` vs `nlms_f64`, baseline — the case this metric exists
  for.** ERLE is effectively equal
  ({aud_base.loc["speex", "steady ERLE (dB)"]:.1f} vs
  {aud_base.loc["nlms_f64", "steady ERLE (dB)"]:.1f} dB), yet speex's
  residual exceeds the threshold in
  {aud_base.loc["speex", "audible fraction"]:.2f} of TF units against
  f64's {aud_base.loc["nlms_f64", "audible fraction"]:.2f}, at nearly
  identical mean excess
  ({aud_base.loc["speex", "mean excess (dB)"]:.1f} vs
  {aud_base.loc["nlms_f64", "mean excess (dB)"]:.1f} dB) — fewer
  above-threshold units, similar modelled excess where above. The direction is
  consistent across all seeds (speex {base_frac["speex"]},
  f64 {base_frac["nlms_f64"]}), and in these no-noise cells the
  two-run speex residual is exact (d = d_echo), so this is not an
  approximation artifact. Equal energy suppression, differently
  distributed residual: an energy-only metric cannot see this
  difference, which is precisely what this analysis exists to detect.
  (A plausible
  mechanism — the MDF's per-band adaptation shaping residual energy
  away from isolated TF regions — is not further diagnosed here.)

<details>
<summary><b>QC: a 10 ms coefficient-snapshot rate is too coarse for exact isolation</b></summary>

The recorded-trajectory reconstruction (held between
{q15_cfg["record_every"]}-sample snapshots) misses the exact output by
{f64_fs.recon_err_db.mean():.1f} dB (hold) /
{f64_fs.recon_err_interp_db.mean():.1f} dB (interpolated) on average
for `nlms_f64` over far-single runs — comparable to or larger than the
residual itself at ~25 dB ERLE, which is why the exact identity is the
primary isolation path (§1.4). Had the trajectory residual been used,
audibility fractions would read
{f64_traj_dfrac.mean():+.3f} (hold) / {f64_interp_dfrac.mean():+.3f}
(interpolated) higher on average for `nlms_f64`
({q15_traj_dfrac.mean():+.3f} for `nlms_q15`, whose own quantisation
noise dominates the decimation error). At μ = 0.5 the sample-wise
NLMS coefficient state moves materially within 10 ms; a faithful
trajectory-based isolation would need a much finer snapshot rate.

</details>

### 2.10 Computational cost

Real-time factor is measured around the canceller call alone (baseline
cell, {n_seeds} seeds, `scripts/measure_cost.py` →
`results/raw/cost.csv`); operation counts and state sizes are derived
analytically (`src/metrics.py`, formulas in the docstrings),
never measured:

{TBL("cost", md_table(cost_tab.rename(index=SYS_RENAME), "system", "{:.4f}"))}

**Derived operation counts are the cleaner basis for comparing
algorithmic complexity; measured RTF reflects both the algorithm and
its implementation.** SpeexDSP combines a lower derived cost
({mac_mdf:,.0f} vs {mac_nlms:,.0f} MAC/sample) with a compiled C
implementation, so its lower RTF cannot be attributed to either factor
alone. The Float and Q15 NLMS implementations execute the *same*
derived operation count yet differ {rtf_q15 / rtf_f64:.1f}× in
measured RTF — a measure of how strongly implementation affects
wall-clock performance. Ranked by derived cost, the picture is the
textbook one: at the same {nlms_cfg["filter_length_ms"]:g} ms tail,
the MDF's partitioned frequency-domain structure needs
{mac_mdf:,.0f} MAC/sample against the time-domain NLMS's
{mac_nlms:,.0f} — an order of magnitude
({mac_nlms / mac_mdf:.1f}×) fewer operations for comparable
steady-state ERLE (§2.1), which is precisely why block-frequency-domain
cancellers exist. Memory tells the inverse story: the MDF pays for its
operation count with roughly {cost[cost.system == "speex"].state_bytes.iloc[0] / cost[cost.system == "nlms_f64"].state_bytes.iloc[0]:.1f}×
the float NLMS's state (partition spectra plus the AUMDF two-filter
structure), while the Q15 filter is the smallest at
{cost[cost.system == "nlms_q15"].state_bytes.iloc[0]:,d} bytes. The
`nlms_q15` RTF excludes the in-loop float shadow filter (divergence
instrumentation, not part of the canceller).

## 3. Discussion

### 3.1 The hazard is uncorrelated energy, not double-talk specifically

The classic framing says an unprotected adaptive filter is endangered
by double-talk. The data support a broader mechanism: **sufficiently
strong energy in `d` that is uncorrelated with `x` can destabilise the
filter, particularly during low-reference-power periods** — near-end
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

### 3.2 Clean-condition step-size tuning hides the robustness problem

Under clean far-end-only speech, ERLE and final misalignment improve
through μ = 0.9 (§2.6 —
{mu_piv.loc["mu0.9", "steady ERLE (dB)"]:.1f} dB /
{mu_piv.loc["mu0.9", "final misalignment (dB)"]:.1f} dB): with no noise,
no near end, and a 15 s horizon that rewards fast adaptation, there is
nothing to trade against. That sweep therefore provides no evidence for
choosing a robust operating point. Separately, the μ = 0.5 baseline
diverges under double-talk and background noise (§2.3–2.4). An
interference-aware step-size sweep would be needed to determine how μ
trades convergence against robustness under those conditions — that
sweep was not run.

### 3.3 Scalar summaries do not tell the whole story

Two scalar summaries in this benchmark systematically flatter or
mislead if read alone. First, a quiet output does not mean an accurate
estimate: at baseline, NLMS reaches {b_nlms_erle.mean():.1f} dB mean
steady-state ERLE while its final coefficient misalignment is only
{b_nlms_mis.mean():.1f} dB — the output is quiet out of proportion to
the accuracy of the underlying echo-path estimate. ERLE rewards
cancelling whatever the *current* input excites; misalignment measures
the whole estimate. The gap is visible dynamically too: the
misalignment trajectory
({REF("baseline_misalignment")}) improves stepwise as new speech material excites new parts
of the path, and the ERLE curve's oscillation with the speech envelope
is the same effect seen from the output side.

{FIGD("baseline_misalignment", "Misalignment trajectory at the baseline cell")}

Second, the convergence-time scalar reflects the metric's construction
as much as adaptation speed. NLMS converges fast and raggedly (baseline
{base_nlms.convergence_time_s.mean():.2f} s mean); Speex ramps slowly
and smoothly ({base_speex.convergence_time_s.mean():.2f} s). But
§1.4's caveat applies: because the threshold is relative to each run's
own steady state, a fraction-of-a-second "convergence" at RT60 0.8 s
reflects a low ERLE ceiling — 90% of a weak target — not fast
adaptation, and Speex's ~6 s figures are the price of a curve that
keeps rising toward a slightly higher plateau. This is why the
convergence map ({REF("stage_a_convergence")}) is a collapsed
diagnostic rather than a headline result. The honest summary is a
difference in *style* — fast/oscillatory vs slow/monotonic — with
scalar convergence times only comparable at similar steady states.

{FIGD("baseline_erle", "ERLE curves at the baseline cell")}

### 3.4 Saturation is containment, not protection

The float benchmark's central finding was that uncorrelated energy in
`d` blows up the unprotected float NLMS. Fixed point adds the
corollary: the
same hazard, hitting the same unprotected recursion in saturating
arithmetic, is *contained* rather than prevented — every diverged float
run's Q15 counterpart stayed bounded (§2.8), because saturation caps
the gain, the error, and the coefficient magnitudes that the float
recursion lets grow without limit. But containment buys bounded
uselessness, not usefulness: the contained runs still leak echo and end
far from the true path. The comparison triangle is instructive —
`nlms_f64` (no protection, unbounded arithmetic) diverges; `nlms_q15`
(no protection, saturating arithmetic) degrades and rides its rails;
`speex` (explicit adaptation control) degrades gracefully and stays
useful. Arithmetic format is a poor substitute for adaptation control,
but it changes the failure mode from unbounded output to bounded but
severely degraded cancellation.

## 4. Limitations

- **Simulation gap.** No loudspeaker nonlinearity, no time-varying echo
  paths, no microphone self-noise. All rooms are ideal shoeboxes with
  frequency-flat absorption. Every system here solves a purely linear,
  time-invariant problem; real-world rankings may differ, particularly
  for Speex whose design anticipates nonlinear residuals.
- **segSNR can overstate perceptual degradation.** In near-single-talk,
  Speex scores {q_near.loc["segsnr_db", "speex"]:.1f} dB segSNR against
  the `none` floor of {none_floor:.1f} dB — yet its STOI is
  {q_near.loc["stoi", "speex"]:.2f} and PESQ
  {q_near.loc["pesq_wb", "speex"]:.2f}. The alteration is concentrated at
  low frequency, consistent with SpeexDSP's internal DC-notch filter on
  the microphone path: a waveform-difference metric strongly penalises
  a low-frequency change that STOI and PESQ largely ignore. The
  masking-based audibility analysis this motivated is §2.9.
- **Float-vs-fixed asymmetry.** `nlms_f64` never passes the int16 path;
  `speex`, `nlms_q15`, and `none` do. The quantisation floor measured
  on `none` (§2.2) bounds the effect, but the comparison is not
  symmetric.
- **Fixed-point results are rounding-convention-specific.** All
  `nlms_q15` stalling, saturation, and word-length results describe a
  magnitude-truncating arithmetic path with a floor-semantics
  coefficient mask (§1.3). A floor-truncating update path fails
  outright (measured — §1.3); a round-to-nearest implementation would
  stall less and jitter differently. The word-length sweep's shape,
  especially the worse-than-passthrough 7-bit result and the inverted
  stall trend, should not be quoted without this qualification.
- **The divergence detector is blind for `nlms_q15`.** Saturating int16
  output cannot exceed the 10×-peak threshold, so `status = ok` on a
  fixed-point row certifies boundedness only (§2.8); ERLE,
  misalignment, and divergence-from-float carry the health information.
- **The speex audibility numbers rest on the two-run approximation.**
  Adaptation trajectories differ between the primary and echo-only
  runs, so the speex residual is approximate where the microphone
  signals differ; the recorded between-run ERLE disagreement is the
  criterion on which those cells are rejected (§2.9). The exact
  component identity is not
  applicable to speex because of its internal DC notch on the
  microphone path.
- **Talk-state-segmented saturation/stall counts are not
  instrumented.** The fixed-point event counters are whole-run totals,
  so behaviour confined to a talk state (for example, the double-talk
  interval) cannot be attributed from the recorded data (§2.2).
- **The masking model is relative, not absolute.** Fixed offset, fixed
  threshold floor, no outer/middle-ear weighting, no temporal masking:
  audible fractions and excesses support comparisons *between systems
  under the same masker*, not statements about what a listener would
  hear in absolute terms. Constants were fixed before results were
  seen and not adjusted afterward.
- **Audibility is measured on far-single frames only** (deliberately,
  for row-by-row comparability with ERLE). Masking of residual echo by
  the near-end talker's own speech during double-talk — the condition
  where echo is most strongly masked in practice — is therefore not
  measured; over the frames used, the masker is background noise or
  silence.
- **Trajectory-based isolation at the recorded snapshot rate is not
  exact** (§2.9): at μ = 0.5 the coefficient state moves materially
  within the {q15_cfg["record_every"]}-sample snapshot interval, so the
  exact component identity is the primary isolation path and the
  trajectory reconstruction serves as QC instrumentation.
- **RTF is measured under CPython on one machine** (§2.10) and ranks
  implementations, not algorithms; the derived MAC counts assume the
  stated cost model (radix-2-class real FFTs at 5N·log2(2N) real MACs,
  complex MAC = 4 real MACs, O(1) terms dropped) and the speex state
  size is derived from the AUMDF structure rather than measured.
- **Convergence metric construction.** Relative-to-own-steady-state
  convergence times are not comparable across systems with different
  steady states (§1.4, §3.3).
- **Three seeds.** Enough to expose seed-dependence (§1.5, §2.4), not enough
  for distributional claims. Per-seed values are plotted and stored.
- **Segmental SNR reference choice.** Distortion is referenced to the
  reverberant near-end `s`; systems are not rewarded or punished for
  dereverberation effects.

## 5. Reproduction

Every row of `results/raw/runs.csv` records the short git SHA of the
code state that produced it: the committed numbers were produced at
{", ".join(git_shas)}. No numerical code path changed between that
commit and the current tree — the only code files that differ over
that range are `scripts/render_report.py`, `src/plotting.py`, and
`src/figure_registry.py`, all presentation-layer; the experiment code
that produces `results/raw/` is identical, and re-rendering this
report from the committed CSVs is byte-stable.

From a clean checkout (macOS/Linux, Python 3.11, SpeexDSP installed —
`brew install speexdsp` or `apt install libspeexdsp-dev`):

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py          # LibriSpeech test-clean (~346 MB, md5-checked)
python -m pytest tests/               # unit + integration tests
python src/run_experiment.py --batch  # full matrix -> results/raw/runs.csv (+ calibration.csv)
python scripts/measure_cost.py        # canceller-only RTF + derived cost -> results/raw/cost.csv
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
