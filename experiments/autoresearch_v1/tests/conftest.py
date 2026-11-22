"""Ensure `experiments/autoresearch_v1` root is importable when invoking `pytest` directly."""
import sys
from pathlib import Path

_EXP_ROOT = str(Path(__file__).resolve().parents[1])
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
