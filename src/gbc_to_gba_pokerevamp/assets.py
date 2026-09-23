"""Asset acquisition: shallow-clone pret decompilations and copy front battle sprites locally."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

from gbc_to_gba_pokerevamp.analyze import detect_background
from gbc_to_gba_pokerevamp.paths import data_dir, manifests_dir, project_root, vendor_dir

console = Console()


class BootstrapError(Exception):
    pass


@dataclass
class AssetRecord:
    id: str
    kind: str
    name: str
    game_family: str
    generation: int
    view: str
    source_repo: str
    source_commit: str
    source_path: str
    local_path: str
    width: int
    height: int
    mode: str
    opaque_color_count: int
    frame_cropped: bool


def load_sources() -> dict:
    path = manifests_dir() / "sources.json"
    if not path.exists():
        raise BootstrapError(f"Missing static manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git_available() -> bool:
    return shutil.which("git") is not None


def ensure_repo(name: str, url: str, update: bool) -> str:
    dest = vendor_dir() / name
    if not dest.exists():
        if not git_available():
            raise BootstrapError(
                "Git is required for automatic asset acquisition but was not found on PATH.\n"
                "Install Git, or clone these manually into vendor/:\n"
                "  https://github.com/pret/pokecrystal.git\n"
                "  https://github.com/pret/pokefirered.git\n"
                "  https://github.com/pret/pokeemerald.git"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Cloning[/] {url} -> {dest} (depth 1)")
        result = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], capture_output=True, text=True)
        if result.returncode != 0:
            raise BootstrapError(f"Upstream clone failed for {url}:\n{result.stderr.strip()}")
    elif update and git_available():
        console.print(f"[cyan]Updating[/] {dest}")
        result = subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[yellow]warning:[/] pull failed for {name}: {result.stderr.strip()}")
    else:
        console.print(f"[green]Found[/] {dest}")
    if git_available():
        result = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return "unknown"


def canonicalize(name: str, aliases: dict[str, str], strip_suffixes: list[str] | None = None) -> str:
    n = name.lower().strip()
    for suf in strip_suffixes or []:
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    n = re.sub(r"_+", "_", n)
    return aliases.get(n, n)


def _name_from_path(repo_root: Path, path: Path, kind: str) -> str:
    rel = path.relative_to(repo_root)
    parts = list(rel.parts)
    if kind == "pokemon":
        # gfx/pokemon/front/abra.png -> abra ; graphics/pokemon/unown/a/front.png -> unown_a
        if not path.stem.startswith("front"):
            return path.stem
        start = parts.index("pokemon") + 1 if "pokemon" in parts else 0
        dirs = [p for p in parts[start:-1] if p != "front"]
        return "_".join(dirs)
    return path.stem


def discover(repo_root: Path, globs: list[str], kind: str, sources: dict) -> list[tuple[Path, str]]:
    excl = sources["exclude_name_parts"]
    aliases = sources["name_aliases"]
    strip = sources["trainer_strip_suffixes"] if kind == "trainer" else None
    found: dict[str, Path] = {}
    for g in globs:
        for p in sorted(repo_root.glob(g)):
            raw = _name_from_path(repo_root, p, kind)
            lowered = raw.lower()
            if any(part in lowered.split("_") or lowered.startswith(part) for part in excl):
                continue
            name = canonicalize(raw, aliases, strip)
            if not name or name in found:
                continue
            found[name] = p
    return sorted(found.items(), key=lambda kv: kv[0])


_RGB_LINE = re.compile(r"RGB\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*;\s*(PAL_\w+)")
_MAP_LINE = re.compile(r"db\s+(PAL_\w+)\s*;\s*(\w+)")


def load_gen1_palettes(repo_root: Path, spec: dict, aliases: dict[str, str]) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, str]]:
    """Parse Red/Blue/Yellow SGB palettes: PAL name -> 4 RGB colors, species -> PAL name.

    Gen 1 sprites are stored as 4-shade grayscale; the game colors them per species using
    these tables, so applying them here gives the sprites the colors players actually saw.
    """
    cfg = spec.get("gen1_palettes")
    if not cfg:
        return {}, {}
    table_path = repo_root / cfg["table"]
    map_path = repo_root / cfg["map"]
    if not table_path.exists() or not map_path.exists():
        console.print(f"[yellow]warning:[/] Gen 1 palette files not found in {repo_root}; sprites stay grayscale")
        return {}, {}
    palettes: dict[str, list[tuple[int, int, int]]] = {}
    for m in _RGB_LINE.finditer(table_path.read_text(encoding="utf-8", errors="replace")):
        vals = [int(v) for v in m.groups()[:12]]
        name = m.group(13)
        colors = [tuple(round(c * 255 / 31) for c in vals[i : i + 3]) for i in range(0, 12, 3)]
        palettes.setdefault(name, colors)  # first definition wins (IF DEF(_RED) blocks)
    species: dict[str, str] = {}
    for m in _MAP_LINE.finditer(map_path.read_text(encoding="utf-8", errors="replace")):
        species[canonicalize(m.group(2), aliases)] = m.group(1)
    return palettes, species


def colorize_gen1(img: Image.Image, colors: list[tuple[int, int, int]]) -> Image.Image:
    """Map a 4-shade grayscale sprite onto a 4-color SGB palette (lightest -> colors[0])."""
    gray = np.array(img.convert("L"))
    levels = np.unique(gray)
    if len(levels) <= 4:
        lut = {int(v): 3 - i for i, v in enumerate(sorted(levels))}  # darkest -> index 3
        if len(levels) < 4:
            lut = {int(v): int(round((255 - v) / 85)) for v in levels}
        idx = np.vectorize(lut.get)(gray)
    else:
        idx = np.clip(np.round((255 - gray.astype(np.int32)) / 85), 0, 3).astype(np.int32)
    pal = np.array(colors, dtype=np.uint8)
    return Image.fromarray(pal[idx], "RGB")


def prepare_image(path: Path, generation: int, gen1_colors: list[tuple[int, int, int]] | None = None) -> tuple[Image.Image, bool]:
    """Return an RGBA/RGB working copy: first animation frame only, GBA index 0 made transparent."""
    img = Image.open(path)
    img.load()
    cropped = False
    w, h = img.size
    if h > w and h % w == 0:
        img = img.crop((0, 0, w, w))
        cropped = True
    if generation >= 3:
        if img.mode == "P":
            idx = np.array(img)
            border = np.concatenate([idx[0], idx[-1], idx[:, 0], idx[:, -1]])
            if np.mean(border == 0) >= 0.5:
                rgba = np.array(img.convert("RGBA"))
                rgba[..., 3] = np.where(idx == 0, 0, 255).astype(np.uint8)
                rgba[idx == 0, :3] = 0
                return Image.fromarray(rgba, "RGBA"), cropped
        rgba = np.array(img.convert("RGBA"))
        opaque, _, _ = detect_background(rgba, None)
        rgba[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
        rgba[~opaque, :3] = 0
        return Image.fromarray(rgba, "RGBA"), cropped
    if generation == 1 and gen1_colors:
        return colorize_gen1(img, gen1_colors), cropped
    # Gen 1/2: keep the opaque light background so the pipeline exercises background inference.
    return img.convert("RGB"), cropped


def _find_pal(path: Path, name: str) -> Path | None:
    for d in (path.parent, path.parent.parent):
        cand = d / name
        if cand.exists():
            return cand
    return None


def _read_jasc_pal(path: Path) -> list[tuple[int, int, int]] | None:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if len(lines) < 3 or lines[0] != "JASC-PAL":
        return None
    try:
        n = int(lines[2])
        colors = [tuple(int(v) for v in ln.split()[:3]) for ln in lines[3 : 3 + n]]
    except ValueError:
        return None
    return colors if len(colors) == n else None  # type: ignore[return-value]


_RGB2 = re.compile(r"RGB\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")


def shiny_variant(path: Path, generation: int, all_frames: bool = False) -> Image.Image | None:
    """Build the shiny-palette version of a front sprite from the decomp's shiny.pal.

    Gen 3: the indexed PNG's palette is swapped for the 16 JASC entries (same index order).
    Gen 2: shiny.pal holds the two middle colors; white and black are fixed, so the PNG's four
    colors are ranked by lightness and the middle two replaced.
    """
    pal_path = _find_pal(path, "shiny.pal")
    if pal_path is None:
        return None
    img = Image.open(path)
    img.load()
    w, h = img.size
    if h > w and h % w == 0 and not all_frames:
        img = img.crop((0, 0, w, w))
    if generation >= 3:
        if img.mode != "P":
            return None
        colors = _read_jasc_pal(pal_path)
        if not colors:
            return None
        idx = np.array(img)
        pal = np.zeros((256, 3), dtype=np.uint8)
        pal[: len(colors)] = np.array(colors, dtype=np.uint8)
        rgba = np.zeros(idx.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = pal[idx]
        rgba[..., 3] = np.where(idx == 0, 0, 255).astype(np.uint8)
        rgba[idx == 0, :3] = 0
        return Image.fromarray(rgba, "RGBA")
    if generation == 2:
        mids = [tuple(round(int(v) * 255 / 31) for v in m.groups()) for m in _RGB2.finditer(pal_path.read_text(encoding="utf-8", errors="replace"))]
        if len(mids) < 2:
            return None
        rgb = np.array(img.convert("RGB"))
        flat = rgb.reshape(-1, 3)
        uniq = np.unique(flat, axis=0)
        if len(uniq) < 3 or len(uniq) > 4:
            return None
        order = sorted(range(len(uniq)), key=lambda i: -(0.299 * uniq[i][0] + 0.587 * uniq[i][1] + 0.114 * uniq[i][2]))
        # lightest stays (white/background), darkest stays (black), the middle ones go shiny
        mapping = {tuple(int(v) for v in uniq[order[0]]): tuple(int(v) for v in uniq[order[0]])}
        mapping[tuple(int(v) for v in uniq[order[-1]])] = tuple(int(v) for v in uniq[order[-1]])
        middle = order[1:-1]
        for k, i in enumerate(middle):
            mapping[tuple(int(v) for v in uniq[i])] = mids[min(k, 1)]  # type: ignore[assignment]
        out = rgb.copy()
        for src, dst in mapping.items():
            out[np.all(rgb == np.array(src, np.uint8), axis=-1)] = dst
        return Image.fromarray(out, "RGB")
    return None


def _record(img: Image.Image, **kw) -> AssetRecord:
    rgba = np.array(img.convert("RGBA"))
    opaque, _, _ = detect_background(rgba, None)
    n_colors = len(np.unique(rgba[opaque][:, :3], axis=0)) if opaque.any() else 0
    return AssetRecord(width=img.width, height=img.height, mode=img.mode, opaque_color_count=int(n_colors), **kw)


def frame_sheet(path: Path, generation: int, gen1_colors: list[tuple[int, int, int]] | None = None) -> tuple[Image.Image, int] | None:
    """Full vertical stack of animation frames (all frames, uncropped), plus the frame count.

    Crystal keeps every frame in front.png; Emerald keeps two frames in anim_front.png.
    """
    src = path
    if generation >= 3:
        anim = path.with_name("anim_front.png")
        if not anim.exists():
            return None
        src = anim
    img = Image.open(src)
    img.load()
    w, h = img.size
    if h <= w or h % w != 0:
        return None
    n = h // w
    if generation >= 3:
        if img.mode != "P":
            return None
        idx = np.array(img)
        rgba = np.array(img.convert("RGBA"))
        rgba[..., 3] = np.where(idx == 0, 0, 255).astype(np.uint8)
        rgba[idx == 0, :3] = 0
        return Image.fromarray(rgba, "RGBA"), n
    return img.convert("RGB"), n


def bootstrap(update: bool = False, skip_clone: bool = False) -> dict:
    sources = load_sources()
    root = project_root()
    data = data_dir()
    records: list[AssetRecord] = []
    by_kind_game: dict[tuple[str, str], dict[str, str]] = {}

    for repo_name, spec in sources["repositories"].items():
        clone_dir = spec.get("dir", repo_name)
        repo_root = vendor_dir() / clone_dir
        if skip_clone:
            if not repo_root.exists():
                console.print(f"[yellow]skipping[/] {repo_name}: not present in vendor/")
                continue
            commit = "unknown"
        else:
            commit = ensure_repo(clone_dir, spec["url"], update)
        game = spec["game_family"]
        gen = int(spec["generation"])
        era = "gbc" if gen <= 2 else "gba"
        prefix = f"{game}_"
        palettes, species_pal = load_gen1_palettes(repo_root, spec, sources["name_aliases"])
        trainer_pal = palettes.get(spec.get("trainer_palette", ""))
        for kind, globs, sub in (("pokemon", spec["pokemon_globs"], "pokemon"), ("trainer", spec["trainer_globs"], "trainers")):
            items = discover(repo_root, globs, kind, sources)
            out_dir = data / "references" / era / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for name, src in items:
                colors = None
                if gen == 1:
                    colors = palettes.get(species_pal.get(name, "")) if kind == "pokemon" else trainer_pal
                    if colors is None and palettes:
                        colors = palettes.get("PAL_MEWMON") or next(iter(palettes.values()))
                try:
                    img, cropped = prepare_image(src, gen, colors)
                except (OSError, ValueError) as exc:
                    console.print(f"[yellow]skip[/] {src}: {exc}")
                    continue
                if img.width < 16 or img.height < 16 or img.width > 96 or img.height > 96:
                    continue
                dest = out_dir / f"{prefix}{name}.png"
                img.save(dest, format="PNG", optimize=False)
                rec = _record(
                    img,
                    id=f"{kind}:{name}:{game}:front",
                    kind=kind,
                    name=name,
                    game_family=game,
                    generation=gen,
                    view="front",
                    source_repo=f"pret/{clone_dir}",
                    source_commit=commit,
                    source_path=src.relative_to(repo_root).as_posix(),
                    local_path=dest.relative_to(root).as_posix(),
                    frame_cropped=cropped,
                )
                records.append(rec)
                by_kind_game.setdefault((kind, game), {})[name] = rec.local_path
                count += 1
                if kind == "pokemon" and gen >= 2:
                    sheet = frame_sheet(src, gen)
                    if sheet is not None and sheet[1] > 1:
                        fimg, nframes = sheet
                        fdest = out_dir / f"{prefix}{name}_frames.png"
                        fimg.save(fdest, format="PNG", optimize=False)
                        records.append(_record(
                            fimg, id=f"{kind}:{name}:{game}:frames", kind=kind, name=name, game_family=game, generation=gen,
                            view="frames", source_repo=f"pret/{clone_dir}", source_commit=commit,
                            source_path=src.relative_to(repo_root).as_posix(), local_path=fdest.relative_to(root).as_posix(),
                            frame_cropped=False,
                        ))
                if kind == "pokemon" and gen >= 2:
                    shiny = shiny_variant(src, gen)
                    if shiny is not None:
                        sdest = out_dir / f"{prefix}{name}_shiny.png"
                        shiny.save(sdest, format="PNG", optimize=False)
                        records.append(_record(
                            shiny, id=f"{kind}:{name}:{game}:front_shiny", kind=kind, name=name, game_family=game,
                            generation=gen, view="front_shiny", source_repo=f"pret/{clone_dir}", source_commit=commit,
                            source_path=src.relative_to(repo_root).as_posix() + " + shiny.pal",
                            local_path=sdest.relative_to(root).as_posix(), frame_cropped=cropped,
                        ))
                        anim = src.with_name("anim_front.png")
                        if gen >= 3 and anim.exists():
                            shiny_frames = shiny_variant(anim, gen, all_frames=True)
                            if shiny_frames is not None and shiny_frames.height > shiny_frames.width:
                                shiny_frames.save(out_dir / f"{prefix}{name}_shiny_frames.png", format="PNG", optimize=False)
            console.print(f"  {game}: copied {count} {kind} front sprites")

    manifests = data / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "sources.json").write_text(json.dumps([asdict(r) for r in records], indent=1), encoding="utf-8")

    pokemon_pairs = build_pairs(by_kind_game, "pokemon")
    (manifests / "pokemon_pairs.json").write_text(json.dumps(pokemon_pairs, indent=1), encoding="utf-8")
    trainer_pairs = build_pairs(by_kind_game, "trainer", sources)
    (manifests / "trainer_pairs.json").write_text(json.dumps(trainer_pairs, indent=1), encoding="utf-8")

    sample = build_sample_set(pokemon_pairs, trainer_pairs, sources)
    (data / "sample_set.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")

    summary = {
        "assets": len(records),
        "per_game": {f"{k[1]}:{k[0]}": len(v) for k, v in sorted(by_kind_game.items())},
        "pokemon_pairs": len(pokemon_pairs),
        "trainer_pairs": len(trainer_pairs),
        "sample_pokemon": [s["name"] for s in sample["pokemon"]],
        "sample_trainers": [s["name"] for s in sample["trainers"]],
        "missing_samples": sample["missing"],
    }
    return summary


SOURCE_GAMES = ("rb", "yellow", "gold", "silver", "crystal")


def build_pairs(by_kind_game: dict[tuple[str, str], dict[str, str]], kind: str, sources: dict | None = None) -> list[dict]:
    """One pair per (source game, name) that has a FRLG and/or Emerald counterpart."""
    frlg = by_kind_game.get((kind, "frlg"), {})
    emerald = by_kind_game.get((kind, "emerald"), {})
    pairs: list[dict] = []
    characters = set(sources["trainer_character_names"]) if sources else set()
    for game in SOURCE_GAMES:
        src_map = by_kind_game.get((kind, game), {})
        for name in sorted(src_map):
            targets = []
            if name in frlg:
                targets.append(frlg[name])
            if name in emerald:
                targets.append(emerald[name])
            if not targets:
                continue
            pair = {"name": name, "game": game, "source": src_map[name], "targets": targets}
            if kind == "trainer":
                if name in characters:
                    pair["confidence"] = "medium"
                    pair["match_reason"] = "same named character in both games; Gen III design may differ"
                else:
                    pair["confidence"] = "high"
                    pair["match_reason"] = "identical canonical trainer class name in both games"
            pairs.append(pair)
    return pairs


def build_sample_set(pokemon_pairs: list[dict], trainer_pairs: list[dict], sources: dict, game: str = "crystal") -> dict:
    by_name = {p["name"]: p for p in pokemon_pairs if p["game"] == game}
    tr_by_name = {p["name"]: p for p in trainer_pairs if p["game"] == game}
    missing: list[str] = []
    pokemon = []
    for n in sources["sample_pokemon"]:
        if n in by_name:
            pokemon.append(by_name[n])
        else:
            missing.append(f"pokemon:{n}")
    trainers = []
    for n in sources["sample_trainers"]:
        if n in tr_by_name:
            trainers.append(tr_by_name[n])
        else:
            missing.append(f"trainer:{n}")
    return {"pokemon": pokemon, "trainers": trainers, "missing": missing}
