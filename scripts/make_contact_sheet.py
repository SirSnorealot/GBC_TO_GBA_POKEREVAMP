"""Render a Crystal | revamp | Gen III contact sheet for the sample set (or any pairs manifest).

Usage:
  python scripts/make_contact_sheet.py                       # uses data/sample_set.json, converts as needed
  python scripts/make_contact_sheet.py data/manifests/pokemon_pairs.json --output output/all_pairs.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbc_to_gba_pokerevamp.config import RevampConfig  # noqa: E402
from gbc_to_gba_pokerevamp.contact_sheet import make_contact_sheet  # noqa: E402
from gbc_to_gba_pokerevamp.paths import project_root  # noqa: E402
from gbc_to_gba_pokerevamp.revamp import run_revamp  # noqa: E402


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=str(root / "data" / "sample_set.json"))
    parser.add_argument("--output", default=str(root / "output" / "contact_sheet" / "contact_sheet.png"))
    parser.add_argument("--output-dir", default=str(root / "output" / "contact_sheet"))
    parser.add_argument("--style", default="frlg")
    parser.add_argument("--no-convert", action="store_true", help="reuse existing outputs instead of re-running revamp")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"error: manifest not found: {manifest}. Run the asset bootstrap first.", file=sys.stderr)
        return 1
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = data["pokemon"] + data["trainers"] if isinstance(data, dict) and "pokemon" in data else data
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        src = root / e["source"]
        targets = [root / t for t in e.get("targets", [])]
        ref = next((t for t in targets if t.name.startswith(args.style + "_")), targets[0] if targets else None)
        out = out_dir / src.stem / "revamped.png"
        if not args.no_convert or not out.exists():
            config = RevampConfig(style=args.style, auto_reference=ref is None, compare=False)  # type: ignore[arg-type]
            run_revamp(src, out_dir, config, [ref] if ref else [])
            print(f"converted {src.stem}")
        rows.append((src, out, ref))
    make_contact_sheet(rows, Path(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
