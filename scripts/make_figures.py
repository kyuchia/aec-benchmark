"""Generate every report figure from results/raw/runs.csv.

Usage:
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from plotting import batch_figures  # noqa: E402
from run_experiment import load_config, scenario_key  # noqa: E402


def main() -> None:
    cfg = load_config()
    baseline_cell = (REPO_ROOT / "data" / "generated" / "runs"
                     / scenario_key(cfg["scenario_defaults"]) / "seed0")
    batch_figures(REPO_ROOT / "results" / "raw" / "runs.csv",
                  REPO_ROOT / "results" / "figures",
                  baseline_cell_dir=baseline_cell)
    print("figures written to results/figures/")


if __name__ == "__main__":
    main()
