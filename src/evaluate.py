from __future__ import annotations
"""src/evaluate.py – utilities for log parsing and visualisation."""
import re
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # head-less backend for server environments
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

logger = logging.getLogger("pax.evaluate")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
#                               parsing utilities
# ---------------------------------------------------------------------------

def parse_megatron_log(log_file: Path) -> Dict:
    """Extract validation loss curve from a Megatron-LM training log.

    The regexes are intentionally strict; if we cannot find any datapoint we
    raise to obey the NO-FALLBACK policy in the prompt.
    """
    ppl_pattern = re.compile(r"validation\s+loss\s+=\s+([0-9.]+)")
    step_pattern = re.compile(r"iteration\s+(\d+)")

    curve: List[Dict[str, float]] = []
    with open(log_file, "r", errors="ignore") as fp:
        for ln in fp:
            m_ppl, m_step = ppl_pattern.search(ln), step_pattern.search(ln)
            if m_ppl and m_step:
                curve.append(
                    {"step": int(m_step.group(1)), "val_loss": float(m_ppl.group(1))}
                )

    if not curve:
        raise RuntimeError(f"No metrics parsed from {log_file}; aborting (NO-FALLBACK).")

    return {"val_curve": curve}


# ---------------------------------------------------------------------------
#                               plotting utilities
# ---------------------------------------------------------------------------

def plot_val_curve(metrics: Dict, fig_file: Path, label: str) -> None:
    steps = [m["step"] for m in metrics["val_curve"]]
    losses = [m["val_loss"] for m in metrics["val_curve"]]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label=label, marker="o")

    for x, y in zip(steps, losses):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")

    plt.xlabel("Steps")
    plt.ylabel("Validation Loss (PPL)")
    plt.title(f"Validation Perplexity – {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close()
