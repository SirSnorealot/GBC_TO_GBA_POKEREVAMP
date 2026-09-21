"""Rebuild the pair manifests from already-copied data/references without touching vendor/.

Usage:  python scripts/build_pair_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbc_to_gba_pokerevamp.assets import build_pairs, build_sample_set, load_sources  # noqa: E402
from gbc_to_gba_pokerevamp.paths import data_dir, project_root  # noqa: E402


def main() -> int:
    root = project_root()
    data = data_dir()
    by_kind_game: dict[tuple[str, str], dict[str, str]] = {}
    for era, kinds in (("gbc", ("pokemon", "trainers")), ("gba", ("pokemon", "trainers"))):
        for sub in kinds:
            kind = "pokemon" if sub == "pokemon" else "trainer"
            d = data / "references" / era / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                stem = p.stem
                game = "other"
                for prefix in ("rb_", "yellow_", "gold_", "silver_", "crystal_", "frlg_", "emerald_"):
                    if stem.startswith(prefix):
                        game, stem = prefix[:-1], stem[len(prefix):]
                        break
                by_kind_game.setdefault((kind, game), {})[stem] = p.relative_to(root).as_posix()
    sources = load_sources()
    manifests = data / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    pokemon_pairs = build_pairs(by_kind_game, "pokemon")
    trainer_pairs = build_pairs(by_kind_game, "trainer", sources)
    (manifests / "pokemon_pairs.json").write_text(json.dumps(pokemon_pairs, indent=1), encoding="utf-8")
    (manifests / "trainer_pairs.json").write_text(json.dumps(trainer_pairs, indent=1), encoding="utf-8")
    sample = build_sample_set(pokemon_pairs, trainer_pairs, sources)
    (data / "sample_set.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print(f"pokemon pairs: {len(pokemon_pairs)}, trainer pairs: {len(trainer_pairs)}, missing samples: {sample['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
