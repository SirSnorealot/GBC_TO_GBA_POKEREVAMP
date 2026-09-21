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
    if src_name and canonical_name_from_path(paths[0]) == src_name:
        # Same species: this reference *is* the answer key for colours; ignore style donors.
        styles[0].same_subject = True
        styles[0].adopt_colors = True
        return styles[0], paths[:1]
    if explicit:
        styles[0].adopt_colors = True
    else:
        warnings.append("no same-species reference found; using shape-alike references for shading style only (colours stay the source's)")
    return aggregate_styles(styles, weights), paths


def protected_mask(norm: NormalizedSprite, roles: dict[int, ColorRole], kind: str) -> np.ndarray:
    """Small same-colour components (eyes, mouths, markings) are protected from synthesised shading.

    The area threshold is expressed in source pixels and scaled with the geometry factor.
    """
    max_area = (10 if kind != "trainer" else 6) * max(1.0, norm.scale) ** 2
    prot = np.zeros(norm.mask.shape, dtype=bool)
    for i in roles:
        prot |= small_components(norm.idx == i, int(round(max_area)))
    return prot


def resolve_dark_bodies(
    family_map: np.ndarray,
    level: np.ndarray,
    families: list[Family],
    infos: list[ColorInfo],
    protected: np.ndarray,
    mask: np.ndarray,
) -> int:
    from scipy import ndimage

    from gbc_to_gba_pokerevamp.analyze import EIGHT

    dark = family_map == -2
    interior = erode(dark, 1, connectivity=8) & ~protected  # every 8-neighbour is dark too
    if interior.sum() < 0.03 * mask.sum():
        return 0
    outline_info = next((c for c in infos if c.role == ColorRole.OUTLINE), min(infos, key=lambda c: c.L))
    labels, n = ndimage.label(interior, structure=EIGHT)
    dark_family: Family | None = None
    changed = 0
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() < 12:
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
            if dark_family is None:
                dark_family = Family(id=len(families), colors=[outline_info], base=outline_info, achromatic=True)
                dark_family.levels[outline_info.rgb] = 0
                families.append(dark_family)
            family_map[comp], level[comp] = dark_family.id, 0
        changed += int(comp.sum())
    return changed


def merge_light_accents(family_map: np.ndarray, level: np.ndarray, families: list[Family], protected: np.ndarray) -> int:
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
            if share >= 0.6 and comp.sum() < 0.15 * host_area:
                family_map[comp] = host
                level[comp] = 1
                merged += int(comp.sum())
    return merged


def run_revamp(
    input_path: Path,
    output_root: Path,
    config: RevampConfig,
    reference_paths: list[Path] | None = None,
) -> RevampResult:
    """Convert one sprite. Writes into `<output_root>/<name>/`:

    revamped.png, compare.png (source | revamp | reference), report.json, debug/ (opt).
    """
    warnings: list[str] = []
    kind = infer_kind(input_path, config)
    config = config.model_copy(update={"kind": kind})
    sprite = load_sprite(input_path, kind=kind)  # type: ignore[arg-type]
    warnings.extend(sprite.warnings)

    run_dir = run_dir_for(input_path, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "revamped.png"
    debug_dir: Path | None = None
    if config.debug:
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_rgba_png(sprite.rgba, debug_dir / "01_source_rgba.png")
        save_rgba_png(mask_to_rgba(sprite.opaque_mask), debug_dir / "02_mask.png")

    # Stage: source colour roles.
    infos = analyze_colors(sprite.rgba, sprite.opaque_mask)
    src_idx, src_palette = index_map(sprite.rgba, sprite.opaque_mask)
    roles = {i: c.role for i, c in enumerate(infos)}
    n_src_colors = len(infos)

    # Stage: Gen 1 sprites (Yellow especially) anti-aliased the *outside* of the outline with
    # light pixels; those become ugly dots on any non-white background, so revamps drop them.
    aa = strip_outer_antialias(src_idx, sprite.opaque_mask, infos)
    if aa.any():
        sprite.opaque_mask = sprite.opaque_mask & ~aa
        src_idx = np.where(aa, -1, src_idx)
        ys, xs = np.nonzero(sprite.opaque_mask)
        sprite.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        warnings.append(f"removed {int(aa.sum())} outer anti-aliasing pixels from the silhouette edge")
    if n_src_colors > 12:
        warnings.append(f"source already appears high-colour ({n_src_colors} opaque colours); shading synthesis will be light")
    components = connected_component_count(sprite.opaque_mask)
    if components > 4:
        warnings.append(f"source has {components} disconnected components")
    cw, ch = sprite.content_size
    sw, sh = sprite.size
    if cw >= sw - 1 and ch >= sh - 1:
        warnings.append("subject nearly fills the source canvas; background detection may be unreliable")

    # Stage: reference style (needed before geometry for target occupancy).
    style, used_refs = resolve_reference_style(sprite, config, reference_paths or [], warnings)
    if config.reference_mode == "palette" and used_refs:
        # Palette-only donors: ignore geometry statistics from the reference.
        style.occupancy = 0.0
    target_occ = style.occupancy if used_refs and style.occupancy > 0 else None

    # Stage: geometry normalisation.
    norm = normalize_geometry(src_idx, sprite.bbox, roles, config, target_occ)
    warnings.extend(norm.warnings)
    if norm.scale != 1.0:
        # Pixels removed by a 1px opening are one-pixel-wide structures.
        thin = sprite.opaque_mask & ~dilate(erode(sprite.opaque_mask, 1), 1)
        if thin.any() or norm.scale < 1.0:
            warnings.append(f"very thin structures ({int(thin.sum())} px) may be lost or thickened by scaling")
    if debug_dir:
        save_rgba_png(index_to_rgba(norm.idx, src_palette), debug_dir / "03_normalized.png")
        save_rgba_png(render_role_map(norm.idx, roles), debug_dir / "04_roles.png")

    # Stage: hue families.
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

    protected = protected_mask(norm, roles, kind)

    # Stage: thick black is body, not line (Snorlax's body, Gengar's shadow masses). Interiors
    # of dark regions >= 3px thick become either the deep shade of the colour they sit in or,
    # when they border nothing coloured, a dark body colour of their own. A 1px rim stays line.
    n_body = resolve_dark_bodies(family_map, level, families, infos, protected, norm.mask)
    if n_body:
        warnings.append(f"treated {n_body} thick dark pixels as body/shadow instead of outline")

    # Stage: GBC artists used white for shine on coloured bodies (there was no lighter shade).
    # Small white patches enclosed by one coloured family become that family's light level;
    # large white regions (bellies, wings) stay their own colour.
    n_merged = merge_light_accents(family_map, level, families, protected)
    if n_merged:
        warnings.append(f"treated {n_merged} white highlight pixels as light shades of the surrounding colour")

    # Stage: palette (after the family map is final).
    outline_rgb, ramps = build_palette(infos, families, style, config)
    if debug_dir:
        preview = [outline_rgb] + [c for r in ramps for c in (r.deep, r.shadow, r.base, r.light, r.highlight, r.line)]
        save_rgba_png(palette_preview(preview), debug_dir / "05_palette_preview.png")

    # Stage: resolve GBC checkerboard dithering. Gen III never dithers; a dithered patch was the
    # artist's way of asking for an in-between shade, so give it a solid one.
    n_dither = 0
    for comp, a, b in checker_dither(norm.idx):
        target = comp & ~protected
        if not target.any():
            continue
        light_i, dark_i = (a, b) if infos[a].L >= infos[b].L else (b, a)
        light_fl = color_to_family.get(src_palette[light_i])
        dark_fl = color_to_family.get(src_palette[dark_i])
        if dark_fl is None and light_fl is not None:
            # colour x black -> deep shadow of that colour
            family_map[target], level[target] = light_fl[0], -2
        elif dark_fl is not None and light_fl is not None and dark_fl[0] == light_fl[0]:
            # two shades of one colour -> the lighter shade; shading re-evaluates base pixels anyway
            family_map[target], level[target] = light_fl[0], light_fl[1]
        elif dark_fl is not None and light_fl is not None:
            # colour x white (or another colour) -> a lighter tone of the darker colour
            family_map[target], level[target] = dark_fl[0], min(2, dark_fl[1] + 1)
        else:
            continue
        n_dither += int(target.sum())
    if n_dither:
        warnings.append(f"resolved {n_dither} dithered pixels into solid shades")
    source_dark = family_map == -2

    # Stage: outlines.
    family_L = {f.id: f.base.L for f in families}
    outline = reconstruct_outline(
        norm.mask, source_dark, family_map, family_L, protected, config, style, (config.light_x, config.light_y)
    )
    line = outline.line
    # Dark pixels thinned out of 2px strokes become the owner family's shadow.
    thinned = source_dark & (line == 0)
    if thinned.any():
        owner = outline.owner.copy()
        need = thinned & (owner < 0)
        if need.any():
            owner[need] = nearest_label(family_map, (family_map >= 0) & ~protected)[need]
        family_map[thinned] = owner[thinned]
        level[thinned] = -1
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
        line_family=np.where(outline.owner >= 0, outline.owner, nearest_label(family_map, (family_map >= 0) & ~protected)),
    )
    if debug_dir:
        save_rgba_png(render(state), debug_dir / "06_outlined.png")

    # Stage: shading.
    shade_cfg = config
    if n_src_colors > 12:
        shade_cfg = config.model_copy(update={"shading_strength": config.shading_strength * 0.4, "highlight_strength": 0.0})
    state.level = synthesize_shading(norm.mask, family_map, level, protected, families, shade_cfg, style)
    shaded = render(state)
    if debug_dir:
        save_rgba_png(shaded, debug_dir / "07_shaded.png")
        save_rgba_png(render_level_map(norm.mask, state.level, state.line), debug_dir / "07_levels.png")

    # Stage: cleanup (never touches linework).
    keep = state.line > 0
    cleaned, n_changed = remove_isolated_pixels(shaded, protected, keep, config.cleanup_strength)
    if debug_dir:
        save_rgba_png(cleaned, debug_dir / "08_cleaned.png")

    # Stage: palette enforcement.
    protected_colors = [outline_rgb] + [state.source_palette[i] for i in np.unique(norm.idx[protected]) if i >= 0]
    final, merge_warnings = enforce_color_limit(cleaned, config.max_opaque_colors, protected_colors)
    warnings.extend(merge_warnings)
    if not np.all((final[..., 3] == 0) | (final[..., 3] == 255)):
        raise RevampError("internal error: partially transparent pixels in final output")
    if debug_dir:
        save_rgba_png(final, debug_dir / "09_final.png")

    save_rgba_png(final, output_path)
    if config.indexed_output:
        save_indexed_png(final, run_dir / "revamped_indexed.png", config.max_colors)

    compare_path: Path | None = None
    if config.compare:
        compare_path = run_dir / "compare.png"
        images = [sprite.image, rgba_to_image(final)]
        labels = [f"source: {input_path.name}", "revamp"]
        if used_refs:
            images.append(load_sprite(used_refs[0]).image)
            labels.append(f"reference: {used_refs[0].name}")
        compare_sheet(images, labels, scale=4).save(compare_path, format="PNG")

    opaque = final[..., 3] > 0
    target_colors = np.unique(final[opaque][:, :3], axis=0)
    ys, xs = np.nonzero(opaque)
    report = ConversionReport(
        input=str(input_path),
        output=str(output_path),
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
