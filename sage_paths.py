"""Project-local runtime paths for SAGE.

All SAGE runtime state should live inside the repository by default so a
project can be moved, backed up, or cleaned without chasing files in HOME.
Set SAGE_HOME to override this location explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SAGE_HOME = Path(os.environ.get("SAGE_HOME", PROJECT_ROOT / ".seismicx")).expanduser()
LEGACY_SAGE_HOME = Path.home() / ".seismicx"


def sage_home(*parts: str) -> Path:
    path = SAGE_HOME.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_sage_home() -> Path:
    SAGE_HOME.mkdir(parents=True, exist_ok=True)
    return SAGE_HOME
