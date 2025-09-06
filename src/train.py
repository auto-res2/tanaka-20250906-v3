from __future__ import annotations
"""src/train.py – handles the heavy-lifting for running each training job.
All model training is delegated to Megatron-LM through the Deepspeed
launcher specified in the YAML, therefore no actual model code is hosted
locally.  This file focuses on:
• robustly spawning the subprocess
• collecting logs
• invoking evaluation/plotting utilities
• persisting artefacts in the mandatory .research/iteration1 structure
"""
import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from .evaluate import parse_megatron_log, plot_val_curve

logger = logging.getLogger("pax.train")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
#                           low-level helpers
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: List[str], log_path: Path) -> None:
    """Executes *cmd* while streaming stdout/stderr both to terminal and file.
    Raises RuntimeError if the child terminates with a non-zero exit code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_fp:
        proc = subprocess.Popen(  # noqa: S603,S607 (external call deliberate)
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Live pass-through so the user sees real-time progress
        assert proc.stdout is not None  # for mypy
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fp.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Sub-process {' '.join(cmd)} failed (code {proc.returncode})."
        )


# ---------------------------------------------------------------------------
#                           public API
# ---------------------------------------------------------------------------

def run_experiment(cfg) -> None:  # cfg is ExperimentConfig from main.py
    """Runs a single experiment as described by *cfg*.

    Steps
    1) create directory layout inside .research/iteration1
    2) launch Deepspeed/Megatron-LM with the exact command from YAML
    3) parse resulting train.log for metrics
    4) write <exp>.json + figures
    5) print the JSON to stdout for verification
    """
    print("=" * 80)
    print(f"Experiment: {cfg.name}")
    print(cfg.description)
    print("Launcher command:")
    print(" ", " ".join(cfg.launcher_cmd))
    print("=" * 80)

    # Directory arrangement required by the prompt
    results_root = Path(".research") / "iteration1"
    exp_dir = results_root / cfg.name
    images_dir = results_root / "images"
    exp_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / "train.log"

    # ------------------------------------------------------------------ run
    start_time = datetime.now()
    _run_subprocess(cfg.launcher_cmd, log_file)
    duration = (datetime.now() - start_time).total_seconds()

    # -------------------------------------------------------------- evaluate
    metrics: Dict = parse_megatron_log(log_file)
    metrics["wall_clock_s"] = duration

    result_json_path = results_root / f"{cfg.name}.json"
    with open(result_json_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    # -------------------------------------------------------------- figures
    for fig_spec in cfg.figure_specs:
        if fig_spec["type"] == "val_curve":
            fig_file = images_dir / fig_spec["file"]
            plot_val_curve(metrics, fig_file, label=cfg.name)
            print(f"Generated figure: {fig_file}")

    # -------------------------------------------------------------- stdout
    print("\n--- Results JSON (", cfg.name, ") ------------------------")
    print(json.dumps(metrics, indent=2))
    print("---------------------------------------------------------\n")