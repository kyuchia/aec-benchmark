"""Single source of truth for report figures and tables.

Numbering is DERIVED from registry order and never hand-written:
section "main" figures number Figure 1..N, section "diagnostics" figures
number Figure D1..DN as an independent sequence (so moving a figure
between sections cannot renumber the main sequence or break existing
references), and tables number Table 1..N in a single main sequence.
Every cross-reference in report prose must go through fig_ref() /
table_ref(); the label strings are built here and nowhere else.

Captions are .format() templates: any number inside a caption must
arrive through a placeholder filled from the same aggregate that
produced the figure — never typed by hand.
"""

from __future__ import annotations

# key -> {filename, section, caption template}; dict order is load-bearing.
FIGURES: dict[str, dict] = {
    "stage_a_erle": {
        "filename": "stage_a_erle.png",
        "section": "main",
        "caption": (
            "Steady-state ERLE by RT60 target and loudspeaker–microphone "
            "distance; each cell is the mean over {n_seeds} seeds under "
            "far-end-only speech with no noise, and row labels carry the "
            "achieved RT60 range. Read: Float NLMS and SpeexDSP degrade "
            "monotonically as RT60 increases at every distance; "
            "increasing distance generally reduces ERLE for those two "
            "systems, with small exceptions at the {rt_min:g} s level. "
            "The Q15 NLMS does not follow that pattern: it flattens at the "
            "closest loudspeaker distance "
            "({q15_d03_lo:.1f}–{q15_d03_hi:.1f} dB across the whole RT60 "
            "range) and falls from {q15_a_rt02_d1:.1f} to "
            "{q15_a_rt08_d1:.1f} dB at the middle distance; the "
            "mechanism is explained in §2.8. Far-end-only, noise-free "
            "scenarios only."),
    },
    "stage_b_talk": {
        "filename": "stage_b_talk.png",
        "section": "main",
        "caption": (
            "Talk-state axis at the baseline room: steady-state ERLE over "
            "far-end-only regions (left) and near-end intelligibility as "
            "STOI over near-active regions (right); points are individual "
            "seeds, bars are means, diverged Float NLMS runs are drawn as "
            "crosses. Each panel covers only the talk states where its "
            "metric is defined: ERLE is undefined for near-end-only "
            "speech (no far-end activity, hence no echo to cancel), so "
            "the left panel shows the far-end-only and double-talk "
            "states; STOI is undefined for far-end-only speech (no "
            "near-end speech to assess), so the right panel shows the "
            "double-talk and near-end-only states. The right panel "
            "replaces the earlier segmental-SNR view, whose "
            "near-end-only cells read the shared int16 quantisation "
            "floor rather than any distortion (and whose far-end-only "
            "state had no defined value at all). Double-talk STOI: Passthrough "
            "{stoi_dt_none:.2f}, Float NLMS {stoi_dt_f64:.2f}, Q15 NLMS "
            "{stoi_dt_q15:.2f}, SpeexDSP {stoi_dt_speex:.2f}."),
    },
    "stage_b_noise": {
        "filename": "stage_b_noise.png",
        "section": "main",
        # NOTE: README.md embeds this figure with a hand-maintained
        # caption paraphrasing this text (README is not
        # renderer-generated). Keep the two consistent when editing
        # either side.
        "caption": (
            "Steady-state ERLE under background noise at the baseline "
            "room; points are individual seeds, bars are means, diverged "
            "Float NLMS runs are drawn as crosses and are part of the "
            "data, not excluded. Read: SpeexDSP degrades gracefully as "
            "SNR falls, the unprotected Float NLMS diverges "
            "dose-dependently, and the Q15 NLMS degrades but stays "
            "bounded: mean ERLE falls "
            "{noise_speex_clean:.1f} to {noise_speex_snr10:.1f} dB for "
            "SpeexDSP across the axis, the Float NLMS means "
            "({noise_f64_snr20:.1f} and {noise_f64_snr10:.1f} dB at the "
            "two noise levels) include its diverged runs, and the Q15 "
            "NLMS holds {noise_q15_snr10:.1f} dB at the strongest noise. "
            "Far-end-only speech; no near-end talker."),
    },
    "word_length_sweep": {
        "filename": "word_length_sweep.png",
        "section": "main",
        "caption": (
            "Effective coefficient word length vs steady-state ERLE "
            "(left) and final coefficient misalignment (right) for the "
            "Q15 NLMS with masked coefficient storage; points are "
            "individual seeds, dashed line is the Float NLMS on the same "
            "scenario, dotted line is the no-processing level. Read: a "
            "cliff, not a slope — every masked level lands below the "
            "no-processing line with misalignment far above the true "
            "path. Scope: floor-masked (truncating) coefficient storage; "
            "this does not generalise to other rounding conventions."),
    },
    "audibility": {
        "filename": "audibility.png",
        "section": "main",
        "caption": (
            "Residual-echo audibility at the baseline cell (far-end "
            "only, no noise), per system: fraction of time–frequency "
            "units above the masking threshold (left) and mean excess "
            "over it (right); points are individual seeds, bars are "
            "means; the threshold is the constant floor, since the "
            "masker is silent in this cell. In this cell the SpeexDSP "
            "residual from the two-run isolation is exact, not "
            "approximate: the microphone signal equals the echo alone, "
            "so the echo-only rerun is bit-identical (between-run ERLE "
            "difference {base_speex_two_max:.1f} dB for every seed). "
            "Audible fractions: Passthrough {aud_frac_none:.2f}, Float "
            "NLMS {aud_frac_f64:.2f}, Q15 NLMS {aud_frac_q15:.2f}, "
            "SpeexDSP {aud_frac_speex:.2f}. Baseline cell only; the "
            "measured noise and double-talk audibility cells are "
            "tabulated, not plotted — see the scope note."),
    },
    # ---- diagnostics: independent D-sequence ----
    "baseline_erle": {
        "filename": "baseline_erle.png",
        "section": "diagnostics",
        "caption": (
            "Smoothed ERLE over time at the baseline cell, seed "
            "{diag_seed}; dotted lines mark each run's steady-state "
            "value. Single seed, shown for curve shape, not statistics."),
    },
    "baseline_misalignment": {
        "filename": "baseline_misalignment.png",
        "section": "diagnostics",
        "caption": (
            "Coefficient misalignment against the true echo path at the "
            "baseline cell, seed {diag_seed}: stepwise improvement as "
            "new speech material excites new parts of the path. Single "
            "seed; NLMS systems only (SpeexDSP exposes no coefficients)."),
    },
    "stage_a_convergence": {
        "filename": "stage_a_convergence.png",
        "section": "diagnostics",
        "caption": (
            "Convergence time by RT60 and distance, mean over {n_seeds} "
            "seeds. Read with the convergence-metric caveat: the "
            "threshold is relative to each run's own steady state, so a "
            "weak target reads as fast convergence; cross-system "
            "comparison is only valid at similar steady states."),
    },
    "stage_b_tail": {
        "filename": "stage_b_tail.png",
        "section": "diagnostics",
        "caption": (
            "Steady-state ERLE vs filter (tail) length at the baseline "
            "room; points are individual seeds, bars are means. "
            "Undermodelling costs every system. Both NLMS filters also "
            "lose ground again at the longest tail (Float "
            "{tail_f64_200:.1f} to {tail_f64_400:.1f} dB, Q15 "
            "{tail_q15_200:.1f} to {tail_q15_400:.1f} dB) — consistent "
            "with added gradient noise and slower per-tap adaptation "
            "outweighing the extra modelled reverberation — while "
            "SpeexDSP holds its level there."),
    },
    "stage_b_mu": {
        "filename": "stage_b_mu.png",
        "section": "diagnostics",
        "caption": (
            "NLMS step size vs steady-state ERLE (left) and final "
            "coefficient misalignment (right); points are individual "
            "seeds. SpeexDSP is absent from the misalignment panel "
            "because its coefficients are not observable through the "
            "binding; it appears on the ERLE panel only as the unswept "
            "baseline reference line. Clean far-end-only speech — the "
            "textbook step-size trade-off is absent here."),
    },
    "q15_float_divergence": {
        "filename": "q15_float_divergence.png",
        "section": "diagnostics",
        "caption": (
            "Per-sample coefficient divergence between the Q15 filter "
            "and its float64 shadow on identical quantised input, "
            "baseline cell, seed {diag_seed}. Single seed, clean "
            "conditions; under strong uncorrelated noise the two "
            "trajectories separate immediately instead."),
    },
}

# key -> caption template; dict order derives Table 1..N.
TABLES: dict[str, str] = {
    "calibration": (
        "RT60 calibration provenance per target level: the Sabine-derived "
        "starting absorption, the RT60 it actually achieves, and the "
        "bisection-calibrated absorption used by every run. Read: "
        "uncalibrated Sabine values overshoot systematically, and the "
        "overshoot grows with target RT60."),
    "talk_quality_double": (
        "Near-end quality during double-talk (echo and near-end speech "
        "simultaneously), mean over seeds: segmental SNR and wideband "
        "PESQ against the reverberant near-end reference. STOI for these "
        "states is plotted, not tabulated."),
    "talk_quality_near": (
        "Near-end quality in near-end-only speech (no far-end at all — "
        "pure passthrough), mean over seeds: segmental SNR and wideband "
        "PESQ. The Passthrough row is the measured int16 quantisation "
        "floor of the shared signal path."),
    "divergence_onsets": (
        "Every non-ok run in the matrix (all Float NLMS): divergence "
        "onset time, onset relative to double-talk start where "
        "applicable, and the post-hoc steady-state ERLE of the diverged "
        "output."),
    "tail_convergence": (
        "Convergence time vs filter (tail) length, mean over seeds. "
        "Steady-state ERLE for the same sweep is plotted, not tabulated."),
    "mu_convergence": (
        "Convergence time vs NLMS step size, mean over seeds (Float "
        "NLMS only — the swept parameter does not exist for SpeexDSP). "
        "ERLE and misalignment for this sweep are plotted, not "
        "tabulated."),
    "word_length_diag": (
        "Fixed-point diagnostics per word-length level, mean over seeds "
        "and per full run: convergence time (read with the same "
        "own-steady-state caveat as elsewhere), coefficient divergence "
        "from the float shadow, full-stall events, and gain/coefficient "
        "saturation events. ERLE and misalignment for the sweep are "
        "plotted, not tabulated."),
    "six_cell": (
        "The Q15 counterpart of every diverged Float NLMS run, matched "
        "on axis, level, and seed (identical microphone signal): the "
        "float run's divergence onset and post-hoc ERLE against the Q15 "
        "run's bounded result and saturation activity."),
    "audibility_noise_fraction": (
        "Audible fraction of residual-echo time–frequency units across "
        "the noise axis, mean over seeds. Measured but not plotted: the "
        "SpeexDSP residual in these cells relies on the two-run "
        "approximation — see the scope note."),
    "audibility_noise_excess": (
        "Mean excess above the masking threshold across the noise axis, "
        "mean over seeds. Same scope caveat as the audible-fraction "
        "table."),
    "cost": (
        "Computational cost: measured canceller-only real-time factor "
        "(mean and range over seeds, CPython on one machine) beside the "
        "analytically derived MAC counts and state sizes. The derived "
        "counts provide the cleaner algorithmic comparison; measured RTF "
        "reflects the specific implementations and runtime environment."),
}


def _fig_number(key: str) -> str:
    mains = [k for k, v in FIGURES.items() if v["section"] == "main"]
    diags = [k for k, v in FIGURES.items() if v["section"] == "diagnostics"]
    if key in mains:
        return str(mains.index(key) + 1)
    if key in diags:
        return f"D{diags.index(key) + 1}"
    raise KeyError(f"unknown figure key {key!r}")


def fig_label(key: str) -> str:
    return f"Figure {_fig_number(key)}"


def fig_ref(key: str) -> str:
    """The only sanctioned way to mention a figure in prose."""
    return fig_label(key)


def fig_embed(key: str, ctx: dict) -> str:
    """Markdown image plus its numbered caption."""
    entry = FIGURES[key]
    caption = entry["caption"].format(**ctx)
    return (f"![{fig_label(key)}](figures/{entry['filename']})\n\n"
            f"**{fig_label(key)}.** {caption}")


def table_label(key: str) -> str:
    return f"Table {list(TABLES).index(key) + 1}"


def table_ref(key: str) -> str:
    return table_label(key)


def table_block(key: str, table_md: str, ctx: dict) -> str:
    """Numbered caption line above the rendered markdown table."""
    caption = TABLES[key].format(**ctx)
    return f"**{table_label(key)}.** {caption}\n\n{table_md}"
