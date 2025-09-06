from __future__ import annotations
"""src/main.py – orchestrates the complete experimental workflow."""
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import yaml

from .preprocess import clone_repos, check_dataset_connectivity
from .train import run_experiment

logger = logging.getLogger("pax.main")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CONFIG_PATH = Path("config") / "config.yaml"


# ---------------------------------------------------------------------------
#                               configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    description: str
    launcher_cmd: List[str]
    figure_specs: List[dict]


def _load_config() -> List[ExperimentConfig]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Mandatory configuration file not found at {CONFIG_PATH}.")
    with open(CONFIG_PATH, "r") as fp:
        raw = yaml.safe_load(fp)

    exps: List[ExperimentConfig] = []
    for exp in raw["experiments"]:
        exps.append(
            ExperimentConfig(
                name=exp["name"],
                description=exp["description"],
                launcher_cmd=exp["launcher_cmd"],
                figure_specs=exp["figures"],
            )
        )
    return exps


# ---------------------------------------------------------------------------
#                               entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (orchestration is linear and explicit)
    # 1. Environment sanity-checks ------------------------------------------------
    try:
        check_dataset_connectivity()
        clone_repos()
    except RuntimeError as err:
        print("[FATAL]", err)
        sys.exit(1)

    # 2. Load YAML ----------------------------------------------------------------
    try:
        experiments = _load_config()
    except Exception as err:  # broad catch is fine at the very top level
        print("[FATAL] Unable to load configuration:", err)
        sys.exit(1)

    # 3. Run experiments one-by-one ----------------------------------------------
    for exp_cfg in experiments:
        try:
            run_experiment(exp_cfg)
        except Exception as err:  # noqa: BLE001 – we want to capture *everything*
            print(f"[ABORT] Experiment {exp_cfg.name} terminated – {err}")
            sys.exit(1)

    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
