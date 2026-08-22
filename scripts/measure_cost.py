"""Computational cost -> results/raw/cost.csv.

Real-time factor is measured around the canceller call ALONE — the
batch's per-row wall times include synthesis, metrics, audibility, and
persistence, so they do not isolate the AEC. Measurement uses the
baseline cell's persisted signals (all seeds), one timed pass per
system per seed; the Q15 run disables the float shadow filter (it is
divergence instrumentation, not part of the canceller). MAC counts and
state sizes are derived analytically in src/metrics.py, never measured.

Usage:
    python scripts/measure_cost.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import metrics  # noqa: E402
from aec_nlms import nlms  # noqa: E402
from aec_nlms_fixed import nlms_q15  # noqa: E402
from aec_speex import run_speex_aec  # noqa: E402
from run_experiment import load_config, scenario_key  # noqa: E402
from signals import float_to_int16, int16_to_float  # noqa: E402

FIELDS = ["system", "seed", "proc_s", "audio_s", "rtf",
          "mac_per_sample", "state_bytes"]


def main() -> None:
    cfg = load_config()
    fs = int(cfg["sample_rate"])
    key = scenario_key(cfg["scenario_defaults"])
    n_seeds = int(cfg["speech"]["n_seeds"])
    L = int(round(cfg["systems"]["nlms_f64"]["filter_length_ms"]
                  * 1e-3 * fs))
    frame = int(cfg["systems"]["speex"]["frame_size"])
    mu = float(cfg["systems"]["nlms_f64"]["mu"])
    delta = float(cfg["systems"]["nlms_f64"]["delta"])
    mu_q15 = int(round(mu * 32768))
    delta_q30 = max(1, int(round(delta * (1 << 30))))

    derived = {
        "none": (0.0, 0),
        "nlms_f64": (metrics.nlms_mac_per_sample(L),
                     metrics.nlms_f64_state_bytes(L)),
        "nlms_q15": (metrics.nlms_mac_per_sample(L),
                     metrics.nlms_q15_state_bytes(L)),
        "speex": (metrics.mdf_mac_per_sample(frame, L),
                  metrics.mdf_state_bytes(frame, L)),
    }

    rows = []
    for seed in range(n_seeds):
        cell = REPO_ROOT / "data" / "generated" / "runs" / key / f"seed{seed}"
        z = np.load(cell / "signals.npz")
        scale = json.load(open(cell / "cell_meta.json"))["int16_scale"]
        x, d = z["x"], z["d"]
        audio_s = len(x) / fs
        x16 = float_to_int16(x, scale, name="x")
        d16 = float_to_int16(d, scale, name="d")

        def timed(fn):
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        timings = {
            "none": timed(lambda: int16_to_float(
                float_to_int16(d, scale, name="d"), scale)),
            "nlms_f64": timed(lambda: nlms(x, d, L=L, mu=mu, delta=delta)),
            "nlms_q15": timed(lambda: nlms_q15(
                x16, d16, L=L, mu_q15=mu_q15, delta_q30=delta_q30,
                shadow_float=False)),
            "speex": timed(lambda: run_speex_aec(x16, d16, frame, L, fs)),
        }
        for system, proc_s in timings.items():
            mac, state = derived[system]
            rows.append({
                "system": system, "seed": seed,
                "proc_s": round(proc_s, 4), "audio_s": audio_s,
                "rtf": round(proc_s / audio_s, 5),
                "mac_per_sample": round(mac, 1), "state_bytes": state,
            })
            print(f"seed {seed} {system:>8}: {proc_s:7.3f} s "
                  f"(RTF {proc_s / audio_s:7.4f})")

    out = REPO_ROOT / "results" / "raw" / "cost.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
