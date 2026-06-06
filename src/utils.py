"""Small utility helpers shared by scripts."""

import os
from pathlib import Path


def getenv_path(name: str, default: str) -> Path:
    """Return an environment path as a Path object."""

    return Path(os.environ.get(name, default)).expanduser()
