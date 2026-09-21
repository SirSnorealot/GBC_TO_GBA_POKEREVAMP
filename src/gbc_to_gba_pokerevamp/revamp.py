"""The revamp pipeline: explicit stages that each operate on separate mask/role layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gbc_to_gba_pokerevamp.analyze import analyze_colors, checker_dither, connected_component_count, index_map, strip_outer_antialias
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.geometry import count_isolated_pixels, dilate, erode, nearest_label, remove_isolated_pixels, small_components
from gbc_to_gba_pokerevamp.models import RGB, ColorInfo, ColorRole, ReferenceStyle, SpriteImage
from gbc_to_gba_pokerevamp.normalize import NormalizedSprite, normalize_geometry
from gbc_to_gba_pokerevamp.outline import reconstruct_outline
from gbc_to_gba_pokerevamp.palette import Family, build_palette, enforce_color_limit, group_families, palette_preview
from gbc_to_gba_pokerevamp.recolor import recolor_same_species
from gbc_to_gba_pokerevamp.references import aggregate_styles, canonical_name_from_path, extract_style, rank_references
from gbc_to_gba_pokerevamp.render import RenderState, compare_sheet, index_to_rgba, mask_to_rgba, render, render_level_map, render_role_map
from gbc_to_gba_pokerevamp.report import ConversionReport
from gbc_to_gba_pokerevamp.shading import synthesize_shading
from gbc_to_gba_pokerevamp.sprite_io import load_sprite, save_indexed_png, save_rgba_png


@dataclass
class RevampResult:
    rgba: np.ndarray
    report: ConversionReport
    run_dir: Path
    output_path: Path
    compare_path: Path | None
    report_path: Path | None
    debug_dir: Path | None


class RevampError(Exception):
    pass


def run_dir_for(input_path: Path, output_root: Path) -> Path:
    """Every conversion gets its own folder: output/<sprite name>/."""
    return output_root / input_path.stem


def infer_kind(path: Path, config: RevampConfig) -> str:
    if config.kind != "unknown":
        return config.kind
    parts = {p.lower() for p in path.parts}
    if "trainers" in parts or "trainer" in path.stem.lower():
        return "trainer"
    return "pokemon"


def resolve_reference_style(
    sprite: SpriteImage,
    config: RevampConfig,
    reference_paths: list[Path],
    warnings: list[str],
) -> tuple[ReferenceStyle, list[Path]]:
    """Load explicit references, or rank local Gen III references, and aggregate their style."""
    paths = list(reference_paths)
    explicit = bool(paths)
    if not paths and config.auto_reference:
        ranked = rank_references(sprite, config)
        if not ranked:
            warnings.append("no GBA references available for --auto-reference; run `assets bootstrap`. Falling back to source-expanded palette")
        else:
            paths = [e.path for e, _ in ranked[: max(1, config.reference_count)]]
    if not paths:
        if config.palette_mode != "source-expanded":
            warnings.append("no reference supplied; using neutral Gen III-like shade defaults")
        return ReferenceStyle(), []
    styles: list[ReferenceStyle] = []
    weights: list[float] = []
    for i, p in enumerate(paths):
        ref = load_sprite(p)
        styles.append(extract_style(ref))
        weights.append(1.0 / (1 + i))  # best-ranked reference dominates
    src_name = canonical_name_from_path(sprite.source_path) if sprite.source_path else ""
    if (src_name and canonical_name_from_path(paths[0]) == src_name) or (explicit and config.force_same_subject):
        # Same species: this reference *is* the answer key for colours; ignore style donors.
        styles[0].same_subject = True
        styles[0].adopt_colors = True
        return styles[0], paths[:1]
    if explicit:
        styles[0].adopt_colors = True
    else:
        warnings.append("no same-species reference found; using shape-alike references for shading style only (colours stay the source's)")
    return aggregate_styles(styles, weights), paths


def protected_mask(norm: NormalizedSprite, roles: dict[int, ColorRole], infos: list[ColorInfo], kind: str) -> np.ndarray:
    """Small components that are *accents* (pupils, eye whites, mouths, claws) are protected
    from synthesised shading. A speck only counts as an accent when it is dark, or light and
    ringed by dark linework; stray light pixels in a body region are shading noise and are not.
    """
    from scipy import ndimage

    from gbc_to_gba_pokerevamp.analyze import EIGHT

    max_area = int(round((10 if kind != "trainer" else 6) * max(1.0, norm.scale) ** 2))
    prot = np.zeros(norm.mask.shape, dtype=bool)
    dark = np.zeros(norm.mask.shape, dtype=bool)
    for i, c in enumerate(infos):
        if c.L < 40:
            dark |= norm.idx == i
    for i, c in enumerate(infos):
        comps = small_components(norm.idx == i, max_area)
        if not comps.any():
            continue
        if c.L < 40:
            prot |= comps
            continue
        labels, n = ndimage.label(comps, structure=EIGHT)
        for k in range(1, n + 1):
            comp = labels == k
            ring = dilate(comp, 1, connectivity=8) & ~comp & norm.mask
            if ring.any() and dark[ring].mean() >= 0.5:
                prot |= comp
    return prot


def resolve_dark_bodies(
    family_map: np.ndarray,
    level: np.ndarray,
    families: list[Family],
    infos: list[ColorInfo],
    protected: np.ndarray,
    mask: np.ndarray,
    style: ReferenceStyle | None = None,
) -> int:
    from scipy import ndimage

    from gbc_to_gba_pokerevamp.analyze import EIGHT

    dark = family_map == -2
    interior = erode(dark, 1, connectivity=8) & ~protected  # every 8-neighbour is dark too
    if interior.sum() < 0.03 * mask.sum():
        return 0
    outline_info = next((c for c in infos if c.role == ColorRole.OUTLINE), min(infos, key=lambda c: c.L))
    labels, n = ndimage.label(interior, structure=EIGHT)
    dark_families: dict[int, Family] = {}
    grid = style.family_grid if (style is not None and style.same_subject) else None
    ys_all, xs_all = np.nonzero(mask)
    bx0, by0 = xs_all.min(), ys_all.min()
    bw, bh = max(1, xs_all.max() - bx0), max(1, ys_all.max() - by0)
    changed = 0
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() < 12:
            continue
        # Same species: the official sprite says what sits here. A dark official region there
        # means this black mass is a body part of its own (Cyndaquil's back), not shadow.
        forced: dict | None = None
        if grid is not None and style is not None:
            G = grid.shape[0]
            ys, xs = np.nonzero(comp)
            gx = np.clip(((xs - bx0) / bw * (G - 1)).round().astype(int), 0, G - 1)
            gy = np.clip(((ys - by0) / bh * (G - 1)).round().astype(int), 0, G - 1)
            vals = grid[gy, gx]
            vals = vals[vals >= 0]
            if len(vals):
                idxs, counts = np.unique(vals, return_counts=True)
                rf = next((r for r in style.families if r.get("index") == int(idxs[np.argmax(counts)])), None)
                if rf is not None and rf["base_lch"][0] <= 50 and counts.max() / len(vals) >= 0.5:
                    forced = rf
        if forced is not None:
            key = int(forced["index"])
            fam = dark_families.get(key)
            if fam is None:
                fam = Family(id=len(families), colors=[outline_info], base=outline_info, achromatic=True, forced_ref=forced)
                fam.levels[outline_info.rgb] = 0
                families.append(fam)
                dark_families[key] = fam
            family_map[comp], level[comp] = fam.id, 0
            changed += int(comp.sum())
            continue
        ring = dilate(comp, 3, connectivity=8) & ~dark & mask
        neigh = family_map[ring]
        neigh = neigh[neigh >= 0]
        host: Family | None = None
        if len(neigh):
            vals, counts = np.unique(neigh, return_counts=True)
            cand = next((g for g in families if g.id == int(vals[np.argmax(counts)])), None)
            if cand is not None and not cand.achromatic and counts.max() / len(neigh) >= 0.5:
                host = cand
        if host is not None:
            family_map[comp], level[comp] = host.id, -2
        else:
            fam = dark_families.get(-1)
            if fam is None:
                fam = Family(id=len(families), colors=[outline_info], base=outline_info, achromatic=True)
                fam.levels[outline_info.rgb] = 0
                families.append(fam)
                dark_families[-1] = fam
            family_map[comp], level[comp] = fam.id, 0
        changed += int(comp.sum())
    return changed


def merge_light_accents(family_map: np.ndarray, level: np.ndarray, families: list[Family], protected: np.ndarray, style: ReferenceStyle) -> int:
    """White patches on a coloured body are either shine (GBC had no lighter shade) or a real
    region such as a belly. Shine is small relative to the body it sits on; a belly is not."""
    from scipy import ndimage

    from gbc_to_gba_pokerevamp.analyze import EIGHT

    merged = 0
    for f in families:
        if not f.achromatic or f.base.L < 88:
            continue
        labels, n = ndimage.label((family_map == f.id) & ~protected, structure=EIGHT)
        for i in range(1, n + 1):
            comp = labels == i
            ring = dilate(comp, 1, connectivity=8) & ~comp
            neigh = family_map[ring]
            neigh = neigh[neigh >= 0]
            if len(neigh) == 0:
                continue
            vals, counts = np.unique(neigh, return_counts=True)
            host = int(vals[np.argmax(counts)])
            share = counts.max() / len(neigh)
            host_family = next((g for g in families if g.id == host), None)
            if host_family is None or host_family.achromatic:
                continue
            host_area = int((family_map == host).sum())
            # Shine is small relative to the body it sits on; a belly is not.
            if share >= 0.6 and comp.sum() <= max(12, 0.06 * host_area):
                family_map[comp] = host
                level[comp] = 1
                merged += int(comp.sum())
    return merged


def spatial_split(
    norm: NormalizedSprite,
    infos: list[ColorInfo],
    roles: dict[int, ColorRole],
    families: list[Family],
    family_map: np.ndarray,
    level: np.ndarray,
    dither_mask: np.ndarray,
    protected: np.ndarray,
    style: ReferenceStyle,
) -> int:
    from scipy import ndimage

    from gbc_to_gba_pokerevamp.analyze import EIGHT

    grid = style.family_grid
    if not style.same_subject or grid is None or not style.families:
        return 0
    G = grid.shape[0]
    ys_all, xs_all = np.nonzero(norm.mask)
    bx0, by0 = xs_all.min(), ys_all.min()
    bw, bh = max(1, xs_all.max() - bx0), max(1, ys_all.max() - by0)
    ref_dom = max(style.families, key=lambda rf: rf["pixels"])
    dominant = max(families, key=lambda f: f.pixel_count)
    min_area = int(round(25 * max(1.0, norm.scale) ** 2))
    changed = 0
    new_families: dict[tuple[int, int], Family] = {}
    for ci, info in enumerate(infos):
        if roles[ci] in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE) or info.rgb == dominant.base.rgb:
            continue
        labels, n = ndimage.label((norm.idx == ci) & ~protected, structure=EIGHT)
        for k in range(1, n + 1):
            comp = labels == k
            if comp.sum() < min_area:
                continue
            # Include the dithered mix around this component: it is part of the same feature.
            comp_full = comp | (dither_mask & dilate(comp, 2, connectivity=8))
            ys, xs = np.nonzero(comp_full)
            gx = np.clip(((xs - bx0) / bw * (G - 1)).round().astype(int), 0, G - 1)
            gy = np.clip(((ys - by0) / bh * (G - 1)).round().astype(int), 0, G - 1)
            lookup = np.full(norm.mask.shape, -1, dtype=np.int32)
            lookup[ys, xs] = grid[gy, gx]
            # One GBC colour may span several official regions (flame + belly shading): assign
            # per pixel, but only in coherent patches so the result is not speckled.
            for best in np.unique(lookup[lookup >= 0]):
                rf = next((r for r in style.families if r.get("index") == int(best)), None)
                if rf is None or rf is ref_dom:
                    continue
                if rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"])) < 0.02:
                    continue
                patch = comp_full & (lookup == best)
                patch = ndimage.binary_closing(patch, structure=np.ones((3, 3), bool)) & comp_full
                patch = patch & ~small_components(patch, 7)
                if patch.sum() < 12:
                    continue
                key = (ci, int(best))
                fam = new_families.get(key)
                if fam is None:
                    fam = Family(id=len(families), colors=[info], base=info, achromatic=info.chroma < 12.0, forced_ref=rf)
                    fam.levels[info.rgb] = 0
                    families.append(fam)
                    new_families[key] = fam
                solid = patch & comp
                mixed = patch & ~comp
                family_map[solid] = fam.id
                level[solid] = 0
                family_map[mixed] = fam.id
                level[mixed] = 1 if info.L < 70 else -1  # dither with a lighter/darker partner
                changed += int(patch.sum())
    return changed


def heuristic_regions(
    norm: NormalizedSprite,
    infos: list[ColorInfo],
    roles: dict[int, ColorRole],
    src_palette: list[RGB],
    families: list[Family],
    color_to_family: dict[RGB, tuple[int, int]],
    family_map: np.ndarray,
    level: np.ndarray,
    dither_mask: np.ndarray,
    protected: np.ndarray,
    style: ReferenceStyle,
) -> list[str]:
    """GBC-convention fixups used when no same-species reference is available."""
    warnings: list[str] = []
    # Gen III never dithers; a dithered patch was the artist's way of asking for an in-between
    # shade, so give it a solid one.
    n_dither = 0
    for comp, a, b in checker_dither(norm.idx):
        target = comp & ~protected
        if not target.any():
            continue
        dither_mask |= target
        light_i, dark_i = (a, b) if infos[a].L >= infos[b].L else (b, a)
        light_fl = color_to_family.get(src_palette[light_i])
        dark_fl = color_to_family.get(src_palette[dark_i])
        if dark_fl is None and light_fl is not None:
            ring = dilate(target, 2, connectivity=8) & ~target & norm.mask
            solid_black = (family_map[ring] == -2).mean() if ring.any() else 0.0
            if target.sum() >= 30 or solid_black >= 0.4:
                family_map[target], level[target] = -2, 0
            else:
                family_map[target], level[target] = light_fl[0], -2
        elif dark_fl is not None and light_fl is not None and dark_fl[0] == light_fl[0]:
            family_map[target], level[target] = light_fl[0], light_fl[1]
        elif dark_fl is not None and light_fl is not None:
            family_map[target], level[target] = dark_fl[0], min(2, dark_fl[1] + 1)
        else:
            continue
        n_dither += int(target.sum())
    if n_dither:
        warnings.append(f"resolved {n_dither} dithered pixels into solid shades")
    # Thick black is body, not line (Snorlax's body, Gengar's shadow masses).
    n_body = resolve_dark_bodies(family_map, level, families, infos, protected, norm.mask, style)
    if n_body:
        warnings.append(f"treated {n_body} thick dark pixels as body/shadow instead of outline")
    # GBC artists used white for shine on coloured bodies (there was no lighter shade).
    n_merged = merge_light_accents(family_map, level, families, protected, style)
    if n_merged:
        warnings.append(f"treated {n_merged} white highlight pixels as light shades of the surrounding colour")
    return warnings


@dataclass
class RevampOutcome:
    """Everything a caller (CLI or GUI) needs; no file IO involved."""

    final: np.ndarray
    stages: dict[str, np.ndarray]  # ordered debug stages, name -> RGBA
    sprite: SpriteImage
    infos: list[ColorInfo]
    style: ReferenceStyle
    used_refs: list[Path]
    families: list[Family]
    norm: NormalizedSprite
    report: ConversionReport
    warnings: list[str]


def _parse_rgb(text: str) -> RGB | None:
    parts = text.replace("(", "").replace(")", "").split(",")
    if len(parts) != 3:
        return None
    try:
        r, g, b = (int(p.strip()) for p in parts)
    except ValueError:
        return None
    return (r, g, b)


def apply_color_map(
    config: RevampConfig,
    infos: list[ColorInfo],
    roles: dict[int, ColorRole],
    norm: NormalizedSprite,
    families: list[Family],
    family_map: np.ndarray,
    level: np.ndarray,
    style: ReferenceStyle,
) -> list[str]:
    """Manual overrides from the GUI: force a source colour onto a chosen target colour.

    The target is looked up in the reference's colour families; when found, the source colour
    takes that family (so shading uses the official ramp) at the level of the chosen tone.
    Otherwise the chosen colour becomes the base of a fresh family with synthesised shades.
    """
    notes: list[str] = []
    for src_text, target in config.color_map.items():
        src_rgb = _parse_rgb(src_text)
        if src_rgb is None or target in ("", "auto"):
            continue
        ci = next((i for i, c in enumerate(infos) if c.rgb == src_rgb), None)
        if ci is None:
            continue
        m = norm.idx == ci
        if not m.any():
            continue
        if target == "outline":
            family_map[m] = -2
            notes.append(f"{src_rgb} -> outline")
            continue
        if target == "keep":
            fam = Family(id=len(families), colors=[infos[ci]], base=infos[ci], achromatic=infos[ci].chroma < 12.0)
            fam.levels[src_rgb] = 0
            fam.forced_ref = {"levels": {"0": list(src_rgb)}, "base_lch": [infos[ci].L, infos[ci].chroma, infos[ci].hue],
                              "pixels": 1, "total_pixels": 1, "achromatic": fam.achromatic, "weight": 1.0, "line": None}
            families.append(fam)
            family_map[m], level[m] = fam.id, 0
            notes.append(f"{src_rgb} kept")
            continue
        tgt = _parse_rgb(target)
        if tgt is None:
            continue
        ref_family, lvl = None, 0
        for rf in style.families:
            for k, v in rf["levels"].items():
                if tuple(v) == tgt:
                    ref_family, lvl = rf, int(k)
        base_info = ColorInfo(rgb=tgt, count=int(m.sum()), frequency=0.0, L=0.0, chroma=0.0, hue=0.0, boundary_fraction=0.0, boundary_share=0.0)
        from gbc_to_gba_pokerevamp.colorspace import rgb_to_lch

        L, C, h = (float(v) for v in rgb_to_lch(np.array(tgt, dtype=np.uint8)))
        base_info.L, base_info.chroma, base_info.hue = L, C, h
        fam = Family(id=len(families), colors=[base_info], base=base_info, achromatic=C < 12.0)
        fam.levels[tgt] = 0
        if ref_family is not None:
            fam.forced_ref = ref_family
            fam.level_offset = lvl
        else:
            fam.forced_ref = {"levels": {"0": list(tgt)}, "base_lch": [L, C, h], "pixels": 1, "total_pixels": 1,
                              "achromatic": fam.achromatic, "weight": 1.0, "line": None}
        families.append(fam)
        family_map[m], level[m] = fam.id, 0
        notes.append(f"{src_rgb} -> {tgt}")
    return notes


def revamp_sprite(
    sprite: SpriteImage,
    config: RevampConfig,
    reference_paths: list[Path] | None = None,
    input_name: str = "sprite",
    pinned_pixels: dict[tuple[int, int], RGB] | None = None,
) -> RevampOutcome:
    """Pure pipeline: source sprite + config -> final 64x64 RGBA and all intermediate stages.

    `pinned_pixels` (source coordinates -> colour) are pixels the user painted by hand: the
    automatic recolour and shading leave them alone and they end up exactly that colour.
    """
    warnings: list[str] = list(sprite.warnings)
    kind = config.kind if config.kind != "unknown" else "pokemon"
    stages: dict[str, np.ndarray] = {}
    stages["01_source_rgba"] = sprite.rgba.copy()

    # Manual "transparent" mappings remove colours from the subject before anything else.
    infos = analyze_colors(sprite.rgba, sprite.opaque_mask)
    for src_text, target in config.color_map.items():
        if target != "transparent":
            continue
        rgb = _parse_rgb(src_text)
        if rgb is None:
            continue
        drop = sprite.opaque_mask & np.all(sprite.rgba[..., :3] == np.array(rgb, np.uint8), axis=-1)
        if drop.any():
            sprite.opaque_mask = sprite.opaque_mask & ~drop
            if not sprite.opaque_mask.any():
                raise RevampError("every pixel was mapped to transparent")
            ys, xs = np.nonzero(sprite.opaque_mask)
            sprite.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            warnings.append(f"{int(drop.sum())} pixels of {rgb} made transparent")
    infos = analyze_colors(sprite.rgba, sprite.opaque_mask)
    stages["02_mask"] = mask_to_rgba(sprite.opaque_mask)

    src_idx, src_palette = index_map(sprite.rgba, sprite.opaque_mask)
    roles = {i: c.role for i, c in enumerate(infos)}
    n_src_colors = len(infos)

    # Gen 1 sprites (Yellow especially) anti-aliased the *outside* of the outline with light
    # pixels. Per the revamp rules they are part of the outline.
    aa = strip_outer_antialias(src_idx, sprite.opaque_mask, infos)
    if aa.any():
        outline_idx = next((i for i, c in enumerate(infos) if c.role == ColorRole.OUTLINE), None)
        if outline_idx is None:
            sprite.opaque_mask = sprite.opaque_mask & ~aa
            src_idx = np.where(aa, -1, src_idx)
            ys, xs = np.nonzero(sprite.opaque_mask)
            sprite.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            src_idx = np.where(aa, outline_idx, src_idx)
        warnings.append(f"folded {int(aa.sum())} outer anti-aliasing pixels into the outline")
    if n_src_colors > 12:
        warnings.append(f"source already appears high-colour ({n_src_colors} opaque colours); shading synthesis will be light")
    components = connected_component_count(sprite.opaque_mask)
    if components > 4:
        warnings.append(f"source has {components} disconnected components")

    # Reference style (needed before geometry for target occupancy).
    style, used_refs = resolve_reference_style(sprite, config, reference_paths or [], warnings)
    if config.reference_mode == "palette" and used_refs:
        style.occupancy = 0.0
    target_occ = style.occupancy if used_refs and style.occupancy > 0 else None

    # Geometry normalisation.
    norm = normalize_geometry(src_idx, sprite.bbox, roles, config, target_occ)
    warnings.extend(norm.warnings)
    stages["03_normalized"] = index_to_rgba(norm.idx, src_palette)
    stages["04_roles"] = render_role_map(norm.idx, roles)

    protected = protected_mask(norm, roles, infos, kind)

    # Hand-painted source pixels: locate them on the canvas, protect them from every
    # automatic decision, and remember their exact colour for the end.
    pinned_mask = np.zeros(norm.mask.shape, dtype=bool)
    pinned_rgb = np.zeros(norm.mask.shape + (3,), dtype=np.uint8)
    if pinned_pixels:
        bx, by = sprite.bbox[0], sprite.bbox[1]
        ox, oy = norm.offset
        for (x, y), rgb in pinned_pixels.items():
            cx = ox + int(round((x - bx) * norm.scale))
            cy = oy + int(round((y - by) * norm.scale))
            if 0 <= cx < norm.mask.shape[1] and 0 <= cy < norm.mask.shape[0] and norm.mask[cy, cx]:
                pinned_mask[cy, cx] = True
                pinned_rgb[cy, cx] = rgb
        protected |= pinned_mask
        if pinned_mask.any():
            warnings.append(f"{int(pinned_mask.sum())} hand-painted pixels kept exactly as painted")

    # RECOLOUR FIRST (same-species Pokémon): every body pixel -> an official colour region.
    recolored = (
        recolor_same_species(norm.idx, norm.mask, infos, roles, protected, style)
        if (style.same_subject and kind != "trainer" and config.recolor)
        else None
    )
    if recolored is not None:
        families = recolored.families
        family_map = recolored.family_map
        level = recolored.level
        warnings.append(f"recoloured {recolored.n_pixels} body pixels into {len(families)} official colour regions by position")
    else:
        families = group_families(infos, kind)
        color_to_family: dict[RGB, tuple[int, int]] = {}
        for f in families:
            for rgb, lvl in f.levels.items():
                color_to_family[rgb] = (f.id, lvl)
        family_map = np.full(norm.mask.shape, -1, dtype=np.int32)
        level = np.zeros(norm.mask.shape, dtype=np.int8)
        for i, rgb in enumerate(src_palette):
            m = norm.idx == i
            if roles[i] in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE):
                family_map[m] = -2
            else:
                fid, lvl = color_to_family[rgb]
                family_map[m] = fid
                level[m] = lvl
        dither_mask = np.zeros(norm.mask.shape, dtype=bool)
        warnings.extend(heuristic_regions(norm, infos, roles, src_palette, families, color_to_family, family_map, level, dither_mask, protected, style))
        if not config.recolor and style.same_subject:
            warnings.append("positional recolour disabled; colours matched by hue only")

    # Manual colour overrides win over everything automatic.
    notes = apply_color_map(config, infos, roles, norm, families, family_map, level, style)
    if notes:
        warnings.append("manual colour map: " + ", ".join(notes))

    # Palette.
    ys_all, xs_all = np.nonzero(norm.mask)
    bx0, by0 = xs_all.min(), ys_all.min()
    bw, bh = max(1, xs_all.max() - bx0), max(1, ys_all.max() - by0)
    centroids: dict[int, tuple[float, float]] = {}
    for f in families:
        fy, fx = np.nonzero(family_map == f.id)
        if len(fx):
            centroids[f.id] = (float((fx.mean() - bx0) / bw), float((fy.mean() - by0) / bh))
    outline_rgb, ramps = build_palette(infos, families, style, config, centroids)
    for f in families:
        if f.shade_of is not None:
            m = family_map == f.id
            level[m] = np.clip(level[m] - 1, -2, 2)
            family_map[m] = f.shade_of
    flat = RenderState(
        idx=norm.idx, mask=norm.mask, family_map=family_map, level=level, line=np.zeros(norm.mask.shape, np.int8),
        protected=protected, families=families, outline_rgb=outline_rgb, source_palette=src_palette, source_roles=roles,
    )
    stages["05_recolored"] = render(flat)
    preview = [outline_rgb] + [c for r in ramps for c in (r.deep, r.shadow, r.base, r.light, r.highlight, r.line)]
    stages["05_palette_preview"] = palette_preview(preview)
    source_dark = family_map == -2

    # Outlines.
    family_L = {f.id: f.base.L for f in families}
    if config.rebuild_outline:
        outline = reconstruct_outline(
            norm.mask, source_dark, family_map, family_L, protected, config, style, (config.light_x, config.light_y)
        )
        line = outline.line
        owner_map = outline.owner
        thinned = source_dark & (line == 0)
        if thinned.any():
            owner = outline.owner.copy()
            need = thinned & (owner < 0)
            if need.any():
                owner[need] = nearest_label(family_map, (family_map >= 0) & ~protected)[need]
            family_map[thinned] = owner[thinned]
            level[thinned] = -1
    else:
        from gbc_to_gba_pokerevamp.outline import LINE_BLACK

        line = np.where(source_dark, LINE_BLACK, 0).astype(np.int8)
        owner_map = np.full(norm.mask.shape, -1, dtype=np.int32)
    valid_body = (family_map >= 0) & ~protected
    fallback_owner = nearest_label(family_map, valid_body) if valid_body.any() else family_map
    state = RenderState(
        idx=norm.idx,
        mask=norm.mask,
        family_map=family_map,
        level=level,
        line=line,
        protected=protected,
        families=families,
        outline_rgb=outline_rgb,
        source_palette=src_palette,
        source_roles=roles,
        line_family=np.where(owner_map >= 0, owner_map, fallback_owner),
    )
    stages["06_outlined"] = render(state)

    # Shading.
    if config.shade:
        shade_cfg = config
        if n_src_colors > 12:
            shade_cfg = config.model_copy(update={"shading_strength": config.shading_strength * 0.4, "highlight_strength": 0.0})
        state.level = synthesize_shading(norm.mask, family_map, level, protected, families, shade_cfg, style)
    shaded = render(state)
    stages["07_shaded"] = shaded
    stages["07_levels"] = render_level_map(norm.mask, state.level, state.line)

    # Cleanup (never touches linework).
    keep = state.line > 0
    cleaned, n_changed = remove_isolated_pixels(shaded, protected, keep, config.cleanup_strength)
    if pinned_mask.any():
        cleaned = cleaned.copy()
        cleaned[pinned_mask, :3] = pinned_rgb[pinned_mask]
        cleaned[pinned_mask, 3] = 255
    stages["08_cleaned"] = cleaned

    # Palette enforcement.
    if config.enforce_palette:
        protected_colors = [outline_rgb] + [state.source_palette[i] for i in np.unique(norm.idx[protected]) if i >= 0]
        if pinned_mask.any():
            protected_colors += [tuple(int(v) for v in c) for c in np.unique(pinned_rgb[pinned_mask], axis=0)]  # type: ignore[misc]
        final, merge_warnings = enforce_color_limit(cleaned, config.max_opaque_colors, protected_colors)
        warnings.extend(merge_warnings)
    else:
        final = cleaned
    if not np.all((final[..., 3] == 0) | (final[..., 3] == 255)):
        raise RevampError("internal error: partially transparent pixels in final output")
    stages["09_final"] = final

    opaque = final[..., 3] > 0
    target_colors = np.unique(final[opaque][:, :3], axis=0)
    ys, xs = np.nonzero(opaque)
    report = ConversionReport(
        input=input_name,
        output="",
        kind=kind,
        style=config.style,
        reference_files=[str(p) for p in used_refs],
        source_dimensions=sprite.size,
        source_bbox=sprite.bbox,
        target_dimensions=(final.shape[1], final.shape[0]),
        target_bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        source_opaque_colors=n_src_colors,
        target_opaque_colors=int(len(target_colors)),
        source_palette=[c.rgb for c in infos],
        target_palette=[tuple(int(v) for v in c) for c in target_colors],  # type: ignore[misc]
        source_roles={str(c.rgb): c.role.name for c in infos},
        families=[
            {
                "id": f.id,
                "base": f.base.rgb,
                "achromatic": f.achromatic,
                "pixels": f.pixel_count,
                "levels": {str(k): v for k, v in f.levels.items()},
                "ramp": {k: v for k, v in f.ramp.__dict__.items()} if f.ramp else None,
            }
            for f in families
        ],
        geometry={
            "scale": norm.scale,
            "mode_used": norm.mode_used,
            "offset": norm.offset,
            "content_size": norm.content_size,
            "target_occupancy": target_occ,
        },
        reference_style=style.to_dict(),
        diagnostics={
            "occupied_area_ratio": float(opaque.sum() / opaque.size),
            "outline_ratio": float((state.line > 0).sum() / max(1, opaque.sum())),
            "isolated_pixels": count_isolated_pixels(final),
            "connected_components": connected_component_count(opaque),
            "cleanup_changed_pixels": n_changed,
            "canvas_fit": bool(final.shape[0] == config.canvas_size and final.shape[1] == config.canvas_size),
        },
        config=config.model_dump(),
        warnings=warnings,
    )
    return RevampOutcome(
        final=final, stages=stages, sprite=sprite, infos=infos, style=style, used_refs=used_refs,
        families=families, norm=norm, report=report, warnings=warnings,
    )


def run_revamp(
    input_path: Path,
    output_root: Path,
    config: RevampConfig,
    reference_paths: list[Path] | None = None,
) -> RevampResult:
    """Convert one sprite. Writes into `<output_root>/<name>/`:

    revamped.png, compare.png (source | revamp | reference), report.json, debug/ (opt).
    """
    kind = infer_kind(input_path, config)
    config = config.model_copy(update={"kind": kind})
    sprite = load_sprite(input_path, kind=kind)  # type: ignore[arg-type]
    outcome = revamp_sprite(sprite, config, reference_paths, input_name=str(input_path))

    run_dir = run_dir_for(input_path, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "revamped.png"
    debug_dir: Path | None = None
    if config.debug:
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, img in outcome.stages.items():
            save_rgba_png(img, debug_dir / f"{name}.png")

    final = outcome.final
    save_rgba_png(final, output_path)
    if config.indexed_output:
        save_indexed_png(final, run_dir / "revamped_indexed.png", config.max_colors)

    compare_path: Path | None = None
    if config.compare:
        compare_path = run_dir / "compare.png"
        images = [outcome.sprite.image, rgba_to_image(final)]
        labels = [f"source: {input_path.name}", "revamp"]
        if outcome.used_refs:
            images.append(load_sprite(outcome.used_refs[0]).image)
            labels.append(f"reference: {outcome.used_refs[0].name}")
        compare_sheet(images, labels, scale=4).save(compare_path, format="PNG")

    report = outcome.report
    report.output = str(output_path)
    report_path: Path | None = None
    if config.report:
        report_path = run_dir / "report.json"
        report.write(report_path)
    return RevampResult(
        rgba=final, report=report, run_dir=run_dir, output_path=output_path,
        compare_path=compare_path, report_path=report_path, debug_dir=debug_dir,
    )


def rgba_to_image(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), "RGBA")
