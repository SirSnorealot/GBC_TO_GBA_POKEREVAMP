"""Outline reconstruction following the Gen III revamp conventions.

Rules implemented (from the classic revamping guides):
* never keep the source outlining as-is; rebuild it and keep it one pixel wide,
* most of the outline is a dark shade of the local colour ("outline base"), pure black only
  where the form is in shadow, where different colours meet in shadow, and for solid blobs
  such as pupils and claws,
* the most lit edges take an even lighter "outline highlight" shade,
* no lighter pixels *outside* the outline (they become dots on dark backgrounds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.analyze import exterior_boundary
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.geometry import dilate, erode, illumination, small_components
from gbc_to_gba_pokerevamp.models import ReferenceStyle

LINE_NONE = 0
LINE_BLACK = 1  # darkest palette colour
LINE_BASE = 2  # family outline-base shade (ramp.line)
LINE_HIGHLIGHT = 3  # family outline-highlight shade (ramp.deep)
LINE_EXTREME = 4  # family shadow shade on the very brightest edge pixels (ramp.shadow)

# Backwards-compatible alias used by the cleanup stage ("never touch the silhouette outline").
LINE_EXTERIOR = LINE_BLACK


@dataclass
class OutlineResult:
    line: np.ndarray  # LINE_* codes
    owner: np.ndarray  # family id that colours each line pixel (-1 where n/a)


def _neighbor_families(family_map: np.ndarray) -> np.ndarray:
    """(8, H, W) stack of neighbouring family ids, -1 outside/none."""
    h, w = family_map.shape
    p = np.pad(family_map, 1, constant_values=-1)
    return np.stack([p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])


def _thin_lines(line_px: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Remove line pixels that only thicken a stroke to 2px, keeping connectivity.

    A pixel is removable when it is not on the silhouette edge, has a line neighbour on the
    edge side, and its removal does not disconnect the stroke (Zhang-Suen simple point test).
    """
    out = line_px.copy()
    h, w = out.shape
    inner = out & ~boundary & mask
    ys, xs = np.nonzero(inner)
    order = np.argsort(ys * w + xs)
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if not out[y, x]:
            continue
        ring = []
        for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)):
            yy, xx = y + dy, x + dx
            ring.append(int(out[yy, xx]) if 0 <= yy < h and 0 <= xx < w else 0)
        b = sum(ring)
        a = sum(1 for k in range(8) if ring[k] == 0 and ring[(k + 1) % 8] == 1)
        # Only thin where a 4-neighbour is also a line pixel *and* the stroke is locally 2 wide.
        four = ring[0] + ring[2] + ring[4] + ring[6]
        if four >= 2 and 2 <= b <= 6 and a == 1:
            # Do not erase pixels that separate two different non-line regions (they are real lines).
            out[y, x] = False
    return out


def reconstruct_outline(
    mask: np.ndarray,
    source_dark: np.ndarray,
    family_map: np.ndarray,
    family_L: dict[int, float],
    protected: np.ndarray,
    config: RevampConfig,
    style: ReferenceStyle,
    light: tuple[float, float],
) -> OutlineResult:
    h, w = mask.shape
    boundary = exterior_boundary(mask)

    # 1. Every line pixel: source dark pixels plus a closed silhouette.
    line_px = source_dark.copy()
    if config.outline_strength >= 0.35:
        line_px |= boundary & ~protected
    else:
        gaps = boundary & ~source_dark & ~protected
        line_px |= gaps & dilate(source_dark & boundary, 1, connectivity=8)

    # 2. Solid dark blobs (pupils, claws, thick dark markings) are kept black wholesale.
    blobs = dilate(erode(source_dark, 1, connectivity=8), 1, connectivity=8) & source_dark
    blobs &= ~dilate(boundary, 1)  # a thick dark rim is stroke, not a blob

    # 3. Thin 2px strokes down to 1px (but never blobs or the silhouette edge itself).
    thin_candidates = line_px & ~blobs & ~protected
    thinned = _thin_lines(thin_candidates, mask, boundary)
    removed = thin_candidates & ~thinned
    line_px = line_px & ~removed

    # 4. Owner family for every line pixel: the darker of the neighbouring families, so a stroke
    #    between belly and body is drawn in the body's outline colour.
    neigh = _neighbor_families(np.where(protected, -1, family_map))
    owner = np.full((h, w), -1, dtype=np.int32)
    stack_L = np.full(neigh.shape, np.inf)
    for fid, Lval in family_L.items():
        stack_L[neigh == fid] = Lval
    has_any = np.isfinite(stack_L).any(axis=0)
    darkest = np.argmin(stack_L, axis=0)
    owner_from_neigh = np.take_along_axis(neigh, darkest[None], axis=0)[0]
    owner[line_px & has_any] = owner_from_neigh[line_px & has_any]
    # Line pixels with no coloured neighbour (inside thick strokes) inherit the nearest owner.
    need = line_px & (owner < 0)
    if need.any() and (owner >= 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(~(owner >= 0), return_indices=True)
        owner[need] = owner[iy, ix][need]

    # 5. Tone assignment by illumination of the whole form.
    lit = illumination(mask, mask, light)
    line = np.zeros((h, w), dtype=np.int8)
    line[line_px] = LINE_BASE
    edge_vals = lit[line_px & boundary]
    if len(edge_vals) == 0:
        edge_vals = lit[line_px]
    black_frac = float(np.clip(style.outline_black_fraction * (0.4 + 0.8 * config.outline_strength), 0.1, 0.95))
    hl_frac = float(np.clip(style.lit_outline_fraction * (1.4 - 0.8 * config.outline_strength), 0.0, 0.5))
    t_black = float(np.quantile(edge_vals, black_frac)) if len(edge_vals) else -np.inf
    t_hl = float(np.quantile(edge_vals, 1.0 - hl_frac)) if hl_frac > 0 and len(edge_vals) else np.inf
    t_ext = float(np.quantile(edge_vals, 1.0 - hl_frac * 0.25)) if hl_frac > 0 and len(edge_vals) else np.inf

    black = line_px & (lit <= t_black)
    black |= blobs | (line_px & protected)
    # Strokes between two different colour families stay black unless the form is clearly lit.
    fam_count = np.zeros((h, w), dtype=np.int32)
    for fid in family_L:
        fam_count += np.any(neigh == fid, axis=0)
    if len(edge_vals):
        t_junction = float(np.quantile(edge_vals, min(0.95, black_frac + 0.3)))
        black |= line_px & (fam_count >= 2) & (lit <= t_junction)
    # Lines touching eyes/mouth accents stay black so faces keep their contrast.
    black |= line_px & dilate(protected, 1, connectivity=8)
    line[black] = LINE_BLACK

    highlight = line_px & ~black & (lit >= t_hl)
    highlight &= ~small_components(highlight, 1)
    line[highlight] = LINE_HIGHLIGHT
    extreme = highlight & (lit >= t_ext) & boundary
    extreme &= ~small_components(extreme, 1)
    line[extreme] = LINE_EXTREME

    # Isolated single black pixels inside a base-coloured run read as noise; merge them.
    singles = (line == LINE_BLACK) & small_components(line == LINE_BLACK, 1) & ~blobs & ~protected
    line[singles] = LINE_BASE
    return OutlineResult(line=line, owner=owner)
