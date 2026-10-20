"""The repository root, the base of the default paths into data/, vendor/ and the output directories."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
