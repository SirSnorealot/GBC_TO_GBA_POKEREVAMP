"""Shallow-clone pret decompilations and copy front battle sprites into data/references.

Usage:  python scripts/bootstrap_assets.py [--update] [--skip-clone]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbc_to_gba_pokerevamp.assets import BootstrapError, bootstrap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="git pull --ff-only existing clones")
    parser.add_argument("--skip-clone", action="store_true", help="only index repositories already in vendor/")
    args = parser.parse_args()
    try:
        summary = bootstrap(update=args.update, skip_clone=args.skip_clone)
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
