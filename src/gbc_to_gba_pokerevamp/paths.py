"""Project-relative path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repository root: env override, else walk up from CWD or the package looking for pyproject.toml."""
    env = os.environ.get("POKEREVAMP_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return candidate
    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def vendor_dir() -> Path:
    return project_root() / "vendor"


def manifests_dir() -> Path:
    return project_root() / "assets" / "manifests"
