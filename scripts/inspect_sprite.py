"""Print sprite statistics.  Usage:  python scripts/inspect_sprite.py <path> [--json]"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbc_to_gba_pokerevamp.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "inspect", *sys.argv[1:]]
    app()
