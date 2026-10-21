"""Default external locations; generated artifacts stay outside the source tree."""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "datos"
RESULTS_DIR = WORKSPACE / "resultados"
