from __future__ import annotations
"""src/preprocess.py – prepares the environment (git clones, connectivity)."""
import subprocess
import logging
from pathlib import Path
from typing import Dict

import requests

logger = logging.getLogger("pax.preprocess")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
#                               external repos
# ---------------------------------------------------------------------------

REQUIRED_REPOS: Dict[str, tuple[str, str]] = {
    "Megatron-LM": ("https://github.com/NVIDIA/Megatron-LM.git", "8c07bf6"),
}

EXTERNAL_DIR = Path("external")
EXTERNAL_DIR.mkdir(exist_ok=True)


def clone_repos() -> None:
    """Clones or updates the external repositories at fixed commits."""
    for name, (url, commit) in REQUIRED_REPOS.items():
        repo_dir = EXTERNAL_DIR / name
        try:
            if not repo_dir.exists():
                logger.info(f"Cloning {name} …")
                subprocess.run(["git", "clone", url, str(repo_dir)], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "fetch"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", commit], check=True)
            logger.info(f"{name} ready at commit {commit}.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git operation failed for {name}: {e}") from e


# ---------------------------------------------------------------------------
#                           dataset connectivity check
# ---------------------------------------------------------------------------

def check_dataset_connectivity() -> None:
    """Performs a quick HEAD request to ensure the remote datasets are reachable."""
    test_urls = [
        "https://huggingface.co/datasets/allenai/c4/raw/main/README.md",
        "https://huggingface.co/datasets/tasksource/mmlu/raw/main/README.md",
    ]
    for url in test_urls:
        try:
            r = requests.head(url, timeout=10)
            if r.status_code >= 400:
                raise RuntimeError(f"Dataset URL {url} unreachable (HTTP {r.status_code}).")
        except requests.RequestException as exc:
            raise RuntimeError(f"Connectivity test failed for {url}: {exc}") from exc
