"""Reference discovery, style extraction, and auto-reference ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.analyze import analyze_colors, exterior_boundary, index_map, shape_descriptors
from gbc_to_gba_pokerevamp.colorspace import hue_delta
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.models import RGB, ColorRole, ReferenceStyle, SpriteImage
from gbc_to_gba_pokerevamp.palette import COOL_HUE, WARM_HUE, group_families
from gbc_to_gba_pokerevamp.paths import project_root


@dataclass
class ReferenceEntry:
    path: Path
    name: str
    kind: str
    game: str  # frlg | emerald | other

    def matches_style(self, style: str) -> bool:
        if style == "gen3-mixed":
            return True
        return self.game == style


GAME_PREFIXES = ("rb_", "yellow_", "gold_", "silver_", "crystal_", "frlg_", "emerald_")
GBA_GAMES = ("frlg", "emerald")
FAMILY_GRID = 32


def _game_from_stem(stem: str) -> tuple[str, str]:
    for prefix in GAME_PREFIXES:
        if stem.startswith(prefix):
            return prefix[:-1], stem[len(prefix) :]
    return "other", stem


def shiny_variant(path: Path) -> Path | None:
    """Path of the shiny-palette sibling of a reference (or source) sprite, if it was generated."""
    if path.stem.endswith("_shiny"):
        return path
    cand = path.with_name(path.stem + "_shiny.png")
    return cand if cand.exists() else None


def normal_variant(path: Path) -> Path:
    if path.stem.endswith("_shiny"):
        cand = path.with_name(path.stem[: -len("_shiny")] + ".png")
        return cand if cand.exists() else path
    return path


def list_references(
    kind: str | None = None, game: str | None = None, root: Path | None = None, era: str = "gba", include_shiny: bool = False
) -> list[ReferenceEntry]:
    """List indexed sprites. era='gba' (Gen III references), 'gbc' (Gen I/II sources) or 'all'."""
    root = root or project_root()
    eras = ["gbc", "gba"] if era == "all" else [era]
    entries: list[ReferenceEntry] = []
    for e in eras:
        base = root / "data" / "references" / e
        for k in ("pokemon", "trainers"):
            kind_name = "pokemon" if k == "pokemon" else "trainer"
            if kind and kind != "unknown" and kind_name != kind:
                continue
            d = base / k
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                if p.stem.endswith("_shiny") and not include_shiny:
                    continue
                g, name = _game_from_stem(p.stem)
                if game and game != "gen3-mixed" and g != game:
                    continue
                entries.append(ReferenceEntry(path=p, name=name, kind=kind_name, game=g))
    return entries


def extract_style(sprite: SpriteImage) -> ReferenceStyle:
    """Measure relative shading behavior from one Gen III sprite (never its literal colors)."""
    infos = analyze_colors(sprite.rgba, sprite.opaque_mask)
    style = ReferenceStyle()
    style.reference_paths = [sprite.source_path] if sprite.source_path else []
    style.palette = [c.rgb for c in infos]
    style.opaque_color_count = len(infos)
    x0, y0, x1, y1 = sprite.bbox
    style.occupancy = float(max(x1 - x0, y1 - y0))
    style.mean_L = float(np.average([c.L for c in infos], weights=[c.count for c in infos]))
    style.mean_chroma = float(np.average([c.chroma for c in infos], weights=[c.count for c in infos]))

    outline = next((c for c in infos if c.role == ColorRole.OUTLINE), min(infos, key=lambda c: c.L))
    style.outline_colors = [outline.rgb]
    style.outline_L = outline.L
    style.outline_chroma = outline.chroma
    idx, palette = index_map(sprite.rgba, sprite.opaque_mask)
    boundary = exterior_boundary(sprite.opaque_mask)
    n_boundary = max(1, int(boundary.sum()))
    # Silhouette outline split: black-ish / dark body-color "outline base" / lighter highlight.
    b_ids = idx[boundary]
    b_L = np.array([infos[i].L for i in b_ids])
    black = b_L <= outline.L + 12
    style.outline_black_fraction = float(np.clip(black.mean(), 0.1, 0.9))
    non_black = b_ids[~black]
    if len(non_black):
        # The outline-base tone is the most common non-black edge color; anything clearly
        # lighter than it is an outline highlight.
        vals, counts = np.unique(non_black, return_counts=True)
        base_L = infos[int(vals[np.argmax(counts)])].L
        hl = b_L[~black] > base_L + 12
        style.lit_outline_fraction = float(np.clip(hl.sum() / n_boundary, 0.0, 0.5))
    else:
        style.lit_outline_fraction = 0.0
    boundary_share = {i: float((b_ids == i).mean()) for i in range(len(infos))}
    n_opaque = max(1, int(sprite.opaque_mask.sum()))
    style.outline_fraction = float(sum(c.count for c in infos if c.role in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE)) / n_opaque)

    families = group_families(infos, step_levels=True)
    style.families = []
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    grid = np.full((FAMILY_GRID, FAMILY_GRID), -1, dtype=np.int32)
    for fi, f in enumerate(families):
        by_level: dict[str, tuple[int, int, int]] = {}
        for rgb, lvl in sorted(f.levels.items(), key=lambda kv: -next(c.count for c in f.colors if c.rgb == kv[0])):
            by_level.setdefault(str(lvl), rgb)
        # The family's darkest color that actually runs along the silhouette is its outline base.
        line_candidates = [c for c in sorted(f.colors, key=lambda c: c.L) if boundary_share[infos.index(c)] >= 0.04 and c.L < f.base.L]
        member = np.isin(idx, [infos.index(c) for c in f.colors])
        ys, xs = np.nonzero(member)
        centroid = [float((xs.mean() - x0) / bw), float((ys.mean() - y0) / bh)] if len(xs) else None
        if len(xs):
            gx = np.clip(((xs - x0) / bw * (FAMILY_GRID - 1)).round().astype(int), 0, FAMILY_GRID - 1)
            gy = np.clip(((ys - y0) / bh * (FAMILY_GRID - 1)).round().astype(int), 0, FAMILY_GRID - 1)
            grid[gy, gx] = fi
        style.families.append({
            "index": fi,
            "base_lch": [f.base.L, f.base.chroma, f.base.hue],
            "levels": by_level,
            "line": line_candidates[0].rgb if line_candidates else None,
            "centroid": centroid,
            "pixels": f.pixel_count,
            "total_pixels": n_opaque,
            "achromatic": f.achromatic,
            "weight": 1.0,
        })
    # Lines cut the grid; fill them from the nearest colored cell so lookups always hit a family.
    if (grid >= 0).any():
        _, (giy, gix) = ndimage.distance_transform_edt(grid < 0, return_indices=True)
        grid = grid[giy, gix]
    style.family_grid = grid
    shadow_d, deep_d, light_d, high_d = [], [], [], []
    shadow_cr, light_cr, shadow_hs, light_hs = [], [], [], []
    weights_s, weights_l = [], []
    shadow_px = deep_px = light_px = high_px = fam_px = 0
    for f in families:
        if f.achromatic or len(f.colors) < 2:
            continue
        base = f.base
        darker = sorted([c for c in f.colors if c.L < base.L], key=lambda c: -c.L)
        lighter = sorted([c for c in f.colors if c.L > base.L], key=lambda c: c.L)
        fam_px += f.pixel_count
        shadow_px += sum(c.count for c in f.colors if f.levels[c.rgb] < 0)
        deep_px += sum(c.count for c in f.colors if f.levels[c.rgb] <= -2)
        light_px += sum(c.count for c in f.colors if f.levels[c.rgb] > 0)
        high_px += sum(c.count for c in f.colors if f.levels[c.rgb] >= 2)
        if darker:
            s = darker[0]
            shadow_d.append(base.L - s.L)
            shadow_cr.append(s.chroma / max(1.0, base.chroma))
            # Positive when the shadow hue moved toward the cool target.
            toward = hue_delta(base.hue, COOL_HUE)
            moved = hue_delta(base.hue, s.hue)
            shadow_hs.append(abs(moved) if np.sign(moved) == np.sign(toward) else -abs(moved) * 0.5)
            weights_s.append(f.pixel_count)
            if len(darker) > 1:
                deep_d.append(base.L - darker[-1].L)
        if lighter:
            l = lighter[0]
            light_d.append(l.L - base.L)
            light_cr.append(l.chroma / max(1.0, base.chroma))
            toward = hue_delta(base.hue, WARM_HUE)
            moved = hue_delta(base.hue, l.hue)
            light_hs.append(abs(moved) if np.sign(moved) == np.sign(toward) else -abs(moved) * 0.5)
            weights_l.append(f.pixel_count)
            if len(lighter) > 1:
                high_d.append(lighter[-1].L - base.L)

    def wavg(vals: list[float], w: list[float], default: float, lo: float, hi: float) -> float:
        if not vals:
            return default
        return float(np.clip(np.average(vals, weights=w[: len(vals)] if len(w) == len(vals) else None), lo, hi))

    style.shadow_dL = wavg(shadow_d, weights_s, style.shadow_dL, 6.0, 30.0)
    style.deep_dL = wavg(deep_d, [], style.deep_dL, style.shadow_dL + 6.0, 45.0)
    style.light_dL = wavg(light_d, weights_l, style.light_dL, 5.0, 25.0)
    style.highlight_dL = wavg(high_d, [], style.highlight_dL, style.light_dL + 4.0, 40.0)
    style.shadow_chroma_ratio = wavg(shadow_cr, weights_s, style.shadow_chroma_ratio, 0.6, 1.5)
    style.light_chroma_ratio = wavg(light_cr, weights_l, style.light_chroma_ratio, 0.4, 1.2)
    style.shadow_hue_shift = wavg(shadow_hs, weights_s, style.shadow_hue_shift, 0.0, 30.0)
    style.light_hue_shift = wavg(light_hs, weights_l, style.light_hue_shift, 0.0, 25.0)
    if fam_px > 0:
        style.shadow_fraction = float(np.clip(shadow_px / fam_px, 0.1, 0.5))
        style.deep_fraction = float(np.clip(deep_px / fam_px, 0.0, 0.25))
        style.light_fraction = float(np.clip(light_px / fam_px, 0.05, 0.4))
        style.highlight_fraction = float(np.clip(high_px / fam_px, 0.0, 0.15))
    style.shadow_colors = [c.rgb for f in families for c in f.colors if c.L < f.base.L]
    style.highlight_colors = [c.rgb for f in families for c in f.colors if c.L > f.base.L]
    return style


_NUMERIC = (
    "shadow_dL", "deep_dL", "light_dL", "highlight_dL", "shadow_chroma_ratio", "light_chroma_ratio",
    "shadow_hue_shift", "light_hue_shift", "outline_L", "outline_chroma", "outline_black_fraction", "lit_outline_fraction",
    "mean_chroma", "mean_L", "shadow_fraction", "deep_fraction", "light_fraction", "highlight_fraction",
    "outline_fraction", "occupancy",
)


def aggregate_styles(styles: list[ReferenceStyle], weights: list[float] | None = None) -> ReferenceStyle:
    if not styles:
        return ReferenceStyle()
    if len(styles) == 1:
        return styles[0]
    w = np.array(weights if weights else [1.0] * len(styles), dtype=float)
    w = w / w.sum()
    out = ReferenceStyle()
    for name in _NUMERIC:
        setattr(out, name, float(sum(wi * getattr(s, name) for wi, s in zip(w, styles))))
    out.opaque_color_count = int(round(sum(wi * s.opaque_color_count for wi, s in zip(w, styles))))
    out.palette = [c for s in styles for c in s.palette]
    out.outline_colors = [c for s in styles for c in s.outline_colors]
    out.families = [{**fam, "weight": float(wi)} for wi, s in zip(w, styles) for fam in s.families]
    out.same_subject = styles[0].same_subject
    out.adopt_colors = styles[0].adopt_colors
    out.reference_paths = [p for s in styles for p in s.reference_paths]
    return out


# --- descriptor cache -----------------------------------------------------------------------

_DESC_KEYS = ("aspect", "area_ratio", "compactness", "complexity", "protrusion", "vertical_com", "symmetry", "edge_density")
_DESC_WEIGHTS = np.array([1.2, 1.0, 1.0, 0.8, 0.8, 1.0, 0.6, 0.6])


def _cache_path(root: Path) -> Path:
    return root / "data" / "cache" / "reference_descriptors.json"


def _stamp(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def reference_descriptors(entries: list[ReferenceEntry], root: Path | None = None) -> dict[str, dict[str, float]]:
    """Compute (or load cached) silhouette descriptors for every reference."""
    from gbc_to_gba_pokerevamp.sprite_io import load_sprite

    root = root or project_root()
    cache_file = _cache_path(root)
    cache: dict[str, dict] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    result: dict[str, dict[str, float]] = {}
    dirty = False
    for e in entries:
        key = str(e.path)
        stamp = _stamp(e.path)
        cached = cache.get(key)
        if cached and cached.get("stamp") == stamp:
            result[key] = cached["desc"]
            continue
        sprite = load_sprite(e.path)
        desc = shape_descriptors(sprite.opaque_mask, sprite.rgba)
        cache[key] = {"stamp": stamp, "desc": desc}
        result[key] = desc
        dirty = True
    if dirty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return result


def canonical_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for prefix in GAME_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    for suffix in ("_shiny", "_revamped", "_front", "_crystal", "_gold", "_silver", "_yellow", "_rb", "_gbc", "_gb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def rank_references(sprite: SpriteImage, config: RevampConfig, root: Path | None = None) -> list[tuple[ReferenceEntry, float]]:
    """Rank Gen III references by silhouette similarity; same-species match always ranks first."""
    root = root or project_root()
    kind = config.kind if config.kind != "unknown" else None
    entries = [e for e in list_references(kind=kind, root=root, era="gba") if e.matches_style(config.style)]
    if not entries:
        return []
    descs = reference_descriptors(entries, root)
    src = shape_descriptors(sprite.opaque_mask, sprite.rgba)
    src_vec = np.array([src.get(k, 0.0) for k in _DESC_KEYS])
    mat = np.array([[descs[str(e.path)].get(k, 0.0) for k in _DESC_KEYS] for e in entries])
    scale = mat.std(axis=0) + 1e-6
    dist = np.sqrt((((mat - src_vec) / scale) ** 2 * _DESC_WEIGHTS).sum(axis=1))
    # Hue family similarity matters for palette statistics when the source is chromatic.
    if abs(src.get("hue_x", 0.0)) + abs(src.get("hue_y", 0.0)) > 0.2:
        ref_h = np.array([[descs[str(e.path)].get("hue_x", 0.0), descs[str(e.path)].get("hue_y", 0.0)] for e in entries])
        dist += 0.8 * np.linalg.norm(ref_h - np.array([src["hue_x"], src["hue_y"]]), axis=1)
    name = canonical_name_from_path(sprite.source_path) if sprite.source_path else ""
    scored = list(zip(entries, dist.tolist()))
    scored.sort(key=lambda t: (0 if (name and t[0].name == name) else 1, t[1]))
    return scored
