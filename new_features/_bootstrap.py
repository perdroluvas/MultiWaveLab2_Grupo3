"""Make the project's ``src`` folder importable when running the scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SRC_FOLDER = REPOSITORY_ROOT / "src"
NEW_FEATURES_FOLDER = Path(__file__).resolve().parent


def ensure_import_paths() -> None:
    """Add ``src`` and ``new_features`` to ``sys.path`` once."""
    for folder in (SRC_FOLDER, NEW_FEATURES_FOLDER):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
