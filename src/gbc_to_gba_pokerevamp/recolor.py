"""Recolour stage for same-species conversions: assign every body pixel to an official colour
region *before* any outline or shading work ("switch out the colours" from the revamp guide).

Each source colour (plus dithered mixes and thick black masses, treated as pseudo-colours) is
scored against every reference colour family by
  * position  - where that family sits on the official sprite (blurred layout map), and
  * colour    - similar lightness, similar hue when both are chromatic, white -> white.
The per-pixel winner is smoothed into coherent patches. Within each region the source colours
are ranked by lightness so the main one becomes the region's base tone and the GBC "shading
colour" lands on its shadow tone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.analyze import EIGHT, checker_dither
from gbc_to_gba_pokerevamp.colorspace import hue_delta, rgb_to_lch
from gbc_to_gba_pokerevamp.geometry import dilate, erode, small_components
from gbc_to_gba_pokerevamp.models import RGB, ColorInfo, ColorRole, ReferenceStyle
from gbc_to_gba_pokerevamp.palette import ACHROMATIC_CHROMA, Family


@dataclass
class RecolorResult:
    families: list[Family]
    family_map: np.ndarray  # >=0 family id, -2 line, -1 transparent
    level: np.ndarray
    dither_mask: np.ndarray
    n_pixels: int


def _pseudo_info(rgb: RGB, count: int) -> ColorInfo:
    L, C, h = (float(v) for v in rgb_to_lch(np.array(rgb, dtype=np.uint8)))
    return ColorInfo(rgb=rgb, count=count, frequency=0.0, L=L, chroma=C, hue=h, boundary_fraction=0.0, boundary_share=0.0)


def _layout_maps(style: ReferenceStyle) -> dict[int, np.ndarray]:
    """Blurred, per-cell-normalised probability of each reference family over the ref bbox."""
    grid = style.family_grid
    assert grid is not None
    maps: dict[int, np.ndarray] = {}
    total = np.zeros(grid.shape, dtype=np.float64)
    for rf in style.families:
        ind = (grid == rf["index"]).astype(np.float64)
        if not ind.any():
            continue
        dens = ndimage.gaussian_filter(ind, sigma=1.6, mode="nearest") + 1e-4
        maps[int(rf["index"])] = dens
        total += dens
    for k in maps:
        maps[k] = maps[k] / total
    return maps


def _color_compat(c: ColorInfo, rf: dict, is_dominant_src: bool, ref_has_white: bool) -> float:
    ref_L, ref_C, ref_h = rf["base_lch"]
    share = rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"]))
    achro_c = c.chroma < ACHROMATIC_CHROMA
    achro_r = bool(rf["achromatic"])
    comp = float(np.exp(-((c.L - ref_L) / 26.0) ** 2))
    if achro_c and c.L > 85 and not is_dominant_src:
        # Bellies / markings: the official white only if the sprite really has a white *region*
        # (eye whites alone do not count), else a cream; a saturated body colour only when
        # nothing better exists (that is the "white shine" case).
        if achro_r and ref_L >= 85:
            return 1.0 if share >= 0.02 else 0.25
        if ref_L >= 75 and ref_C <= 65 and 35.0 <= ref_h <= 110.0:
            return 0.85
        return 0.15 if ref_has_white else 0.4
    if achro_c and c.L < 25:
        return 1.0 if ref_L <= 50 else 0.1
    if achro_c or achro_r:
        return 0.6 * comp + 0.2
    d = abs(hue_delta(c.hue, ref_h))
    hue_term = 0.45 + 0.55 * float(np.exp(-((d / 45.0) ** 2)))
    return comp * hue_term + 0.1


def recolor_same_species(
    idx: np.ndarray,
    mask: np.ndarray,
    infos: list[ColorInfo],
    roles: dict[int, ColorRole],
    protected: np.ndarray,
    style: ReferenceStyle,
) -> RecolorResult | None:
    if style.family_grid is None or not style.families:
        return None
    h, w = mask.shape
    ys_all, xs_all = np.nonzero(mask)
    bx0, by0 = xs_all.min(), ys_all.min()
    bw, bh = max(1, xs_all.max() - bx0), max(1, ys_all.max() - by0)
    G = style.family_grid.shape[0]

    # --- pseudo-colour map ---------------------------------------------------------------
    line_role = {i for i, r in roles.items() if r in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE)}
    pcolor = np.full((h, w), -1, dtype=np.int32)
    pinfos: list[ColorInfo] = []
    for i, c in enumerate(infos):
        pinfos.append(c)
        if i not in line_role:
            pcolor[idx == i] = i
    source_dark = np.isin(idx, list(line_role)) if line_role else np.zeros((h, w), dtype=bool)

    dither_mask = np.zeros((h, w), dtype=bool)
    mixes: dict[tuple[int, int], int] = {}
    for comp, a, b in checker_dither(idx):
        target = comp & ~protected
        if not target.any():
            continue
        key = (min(a, b), max(a, b))
        if key not in mixes:
            if a in line_role or b in line_role:
                # Colour x black is how 2bpp drew a dark region; treat it as dark, not as a
                # muddy mid-tone, so it lands on the reference's dark family at that spot.
                mix_rgb = (28, 28, 28)
            else:
                ra, rb_ = np.array(infos[a].rgb), np.array(infos[b].rgb)
                mix_rgb = tuple(int(v) for v in ((ra.astype(int) + rb_.astype(int)) // 2))
            mixes[key] = len(pinfos)
            pinfos.append(_pseudo_info(mix_rgb, 0))  # type: ignore[arg-type]
        pcolor[target] = mixes[key]
        dither_mask |= target
    # Thick black is body, not line: the interior of dark masses becomes a "black" body colour.
    interior = erode(source_dark, 1, connectivity=8) & ~protected
    interior &= ~small_components(interior, 11)
    if interior.any():
        black_idx = len(pinfos)
        dark_info = next((c for c in infos if c.role == ColorRole.OUTLINE), min(infos, key=lambda c: c.L))
        pinfos.append(_pseudo_info(dark_info.rgb, int(interior.sum())))
        pcolor[interior] = black_idx
        source_dark &= ~interior
    for pi in range(len(pinfos)):
        pinfos[pi].count = int((pcolor == pi).sum())

    body = pcolor >= 0
    if not body.any():
        return None
    dominant_src = int(np.bincount(pcolor[body]).argmax())

    # --- scoring -------------------------------------------------------------------------
    layout = _layout_maps(style)
    fam_by_index = {int(rf["index"]): rf for rf in style.families}
    ref_ids = sorted(layout)
    ref_has_white = any(
        rf["achromatic"] and rf["base_lch"][0] >= 85 and rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"])) >= 0.02
        for rf in style.families
    )
    white_ref = next((int(rf["index"]) for rf in style.families if rf["achromatic"] and rf["base_lch"][0] >= 85), None)
    total_ref = max(1.0, float(sum(rf["pixels"] for rf in style.families)))
    ys, xs = np.nonzero(body)
    gx = np.clip(((xs - bx0) / bw * (G - 1)).round().astype(int), 0, G - 1)
    gy = np.clip(((ys - by0) / bh * (G - 1)).round().astype(int), 0, G - 1)
    scores = np.zeros((len(ref_ids), len(ys)), dtype=np.float64)
    pc = pcolor[ys, xs]
    # How much position may override colour: a white belly is white wherever the official pose
    # put it, while a black mass or the body colour follows the official layout closely.
    pos_floor = np.zeros(len(ys))
    for pi in np.unique(pc):
        c = pinfos[pi]
        if c.chroma < ACHROMATIC_CHROMA and c.L > 85 and pi != dominant_src:
            pos_floor[pc == pi] = 0.35
    for ri, rid in enumerate(ref_ids):
        rf = fam_by_index[rid]
        pos = pos_floor + (1.0 - pos_floor) * layout[rid][gy, gx]
        prior = (rf["pixels"] / total_ref) ** 0.25
        compat = np.zeros(len(ys))
        for pi in np.unique(pc):
            compat[pc == pi] = _color_compat(pinfos[pi], rf, pi == dominant_src, ref_has_white)
        scores[ri] = pos * compat * prior
    assign = np.full((h, w), -1, dtype=np.int32)
    assign[ys, xs] = np.array(ref_ids)[scores.argmax(axis=0)]
    # A black mass that lands on a *light* official region (an eye slit, a mouth) is linework,
    # not a body colour: it stays black.
    black_like = [pi for pi in range(len(pinfos)) if pinfos[pi].chroma < ACHROMATIC_CHROMA and pinfos[pi].L < 25]
    if black_like:
        for rid in ref_ids:
            if fam_by_index[rid]["base_lch"][0] > 55:
                back_to_line = np.isin(pcolor, black_like) & (assign == rid) & ~dither_mask
                assign[back_to_line] = -1
                source_dark |= back_to_line
                body &= ~back_to_line
        ys, xs = np.nonzero(body)
        pc = pcolor[ys, xs]
    # Protected light accents (eye whites, teeth) are white in Gen III whenever a white exists.
    if white_ref is not None and white_ref in layout:
        light_prot = protected & body
        for pi in np.unique(pcolor[light_prot]) if light_prot.any() else []:
            if pinfos[pi].chroma < ACHROMATIC_CHROMA and pinfos[pi].L > 85:
                assign[light_prot & (pcolor == pi)] = white_ref

    # --- smoothing into coherent patches (within each pseudo-colour) ---------------------
    kernel = np.ones((5, 5), dtype=np.int32)
    locked = protected & body
    for _ in range(2):
        new_assign = assign.copy()
        for pi in np.unique(pc):
            m = (pcolor == pi) & ~locked
            best_cnt = np.full((h, w), -1, dtype=np.int32)
            for rid in ref_ids:
                cnt = ndimage.convolve(((assign == rid) & m).astype(np.int32), kernel, mode="constant")
                better = m & (cnt > best_cnt)
                new_assign[better] = rid
                best_cnt[better] = cnt[better]
        assign = new_assign
    # Patches too small to read as a region join the neighbouring assignment of the same colour.
    for pi in np.unique(pc):
        m = pcolor == pi
        for rid in ref_ids:
            patch = m & (assign == rid)
            tiny = patch & small_components(patch, 7)
            if not tiny.any():
                continue
            labels, n = ndimage.label(tiny, structure=EIGHT)
            for k in range(1, n + 1):
                comp = labels == k
                ring = dilate(comp, 1, connectivity=8) & ~comp & m
                vals = assign[ring]
                vals = vals[(vals >= 0) & (vals != rid)]
                if len(vals):
                    assign[comp] = np.bincount(vals).argmax()

    # --- families + levels ---------------------------------------------------------------
    families: list[Family] = []
    family_map = np.full((h, w), -1, dtype=np.int32)
    family_map[source_dark] = -2
    level = np.zeros((h, w), dtype=np.int8)
    for rid in ref_ids:
        region = assign == rid
        if not region.any():
            continue
        members = [int(pi) for pi in np.unique(pcolor[region])]
        counts = {pi: int((region & (pcolor == pi)).sum()) for pi in members}
        base_pi = max(members, key=lambda pi: counts[pi])
        base = pinfos[base_pi]
        fam = Family(
            id=len(families), colors=[pinfos[pi] for pi in members], base=base,
            achromatic=base.chroma < ACHROMATIC_CHROMA, forced_ref=fam_by_index[rid],
        )
        lighter = sorted([pi for pi in members if pinfos[pi].L > base.L], key=lambda pi: pinfos[pi].L)
        darker = sorted([pi for pi in members if pinfos[pi].L < base.L], key=lambda pi: -pinfos[pi].L)
        lvl_of = {base_pi: 0}
        for i, pi in enumerate(lighter):
            lvl_of[pi] = min(2, i + 1)
        for i, pi in enumerate(darker):
            lvl_of[pi] = max(-2, -(i + 1))
        for pi, lvl in lvl_of.items():
            fam.levels.setdefault(pinfos[pi].rgb, lvl)
            level[region & (pcolor == pi)] = lvl
        families.append(fam)
        family_map[region] = fam.id
    return RecolorResult(families=families, family_map=family_map, level=level, dither_mask=dither_mask, n_pixels=int(body.sum()))
