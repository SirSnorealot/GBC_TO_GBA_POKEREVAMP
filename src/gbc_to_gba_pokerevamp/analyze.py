"""Background detection, subject masks, and source colour-role analysis."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.colorspace import rgb_to_lch
from gbc_to_gba_pokerevamp.models import RGB, ColorInfo, ColorRole

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
EIGHT = np.ones((3, 3), dtype=bool)


def _border_colors(rgb: np.ndarray) -> Counter:
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0, :], rgb[h - 1, :], rgb[:, 0], rgb[:, w - 1]], axis=0)
    return Counter(tuple(int(v) for v in px) for px in border)


def detect_background(rgba: np.ndarray, indices: np.ndarray | None = None) -> tuple[np.ndarray, RGB | None, list[str]]:
    """Return (opaque_mask, background_color, warnings).

    Alpha wins when present. Otherwise the most frequent border colour is flood-filled
    from the canvas edges so same-coloured enclosed regions stay part of the subject.
    """
    warnings: list[str] = []
    alpha = rgba[..., 3]
    if np.any(alpha < 255):
        if np.any((alpha > 0) & (alpha < 255)):
            warnings.append("source contains partially transparent pixels; thresholded at alpha>=128")
        return alpha >= 128, None, warnings

    rgb = rgba[..., :3]
    counts = _border_colors(rgb)
    total_border = sum(counts.values())
    bg_color, bg_count = counts.most_common(1)[0]
    share = bg_count / max(1, total_border)
    if share < 0.6:
        warnings.append(f"ambiguous background: dominant border colour covers only {share:.0%} of the border")
    # For paletted GBA-style images index 0 is conventionally transparent; use it as a tiebreaker.
    if indices is not None and share < 0.9:
        idx_border = np.concatenate([indices[0, :], indices[-1, :], indices[:, 0], indices[:, -1]])
        if np.mean(idx_border == 0) > share:
            zero_px = rgb[indices == 0]
            if len(zero_px):
                bg_color = tuple(int(v) for v in zero_px[0])  # type: ignore[assignment]

    same = np.all(rgb == np.array(bg_color, dtype=np.uint8), axis=-1)
    labels, n = ndimage.label(same, structure=FOUR)
    if n == 0:
        return np.ones(rgb.shape[:2], dtype=bool), bg_color, warnings  # type: ignore[return-value]
    edge_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    background = np.isin(labels, edge_labels)
    opaque = ~background
    enclosed = same & ~background
    if enclosed.any():
        warnings.append(
            f"{int(enclosed.sum())} interior pixels share the background colour and were kept as subject"
        )
    return opaque, bg_color, warnings  # type: ignore[return-value]


def exterior_boundary(mask: np.ndarray) -> np.ndarray:
    """Opaque pixels with at least one 4-neighbour outside the mask (or on the canvas edge)."""
    padded = np.pad(mask, 1, constant_values=False)
    eroded = ndimage.binary_erosion(padded, structure=FOUR, border_value=0)[1:-1, 1:-1]
    return mask & ~eroded


def index_map(rgba: np.ndarray, opaque: np.ndarray) -> tuple[np.ndarray, list[RGB]]:
    """Map every opaque pixel to an index into a deterministic colour list; background is -1."""
    colors = np.unique(rgba[opaque][:, :3], axis=0)
    palette: list[RGB] = [tuple(int(v) for v in c) for c in colors]  # type: ignore[misc]
    key = rgba[..., 0].astype(np.int64) * 65536 + rgba[..., 1].astype(np.int64) * 256 + rgba[..., 2].astype(np.int64)
    pal_key = colors[:, 0].astype(np.int64) * 65536 + colors[:, 1].astype(np.int64) * 256 + colors[:, 2].astype(np.int64)
    order = np.argsort(pal_key)
    pos = np.searchsorted(pal_key[order], key)
    pos = np.clip(pos, 0, len(order) - 1)
    idx = order[pos]
    idx = np.where(opaque, idx, -1).astype(np.int32)
    return idx, palette


def analyze_colors(rgba: np.ndarray, opaque: np.ndarray) -> list[ColorInfo]:
    """Compute per-colour statistics and assign tentative roles."""
    idx, palette = index_map(rgba, opaque)
    boundary = exterior_boundary(opaque)
    n_opaque = int(opaque.sum())
    n_boundary = max(1, int(boundary.sum()))
    lch = rgb_to_lch(np.array(palette, dtype=np.uint8))
    infos: list[ColorInfo] = []
    for i, rgb in enumerate(palette):
        pix = idx == i
        count = int(pix.sum())
        on_boundary = int((pix & boundary).sum())
        infos.append(
            ColorInfo(
                rgb=rgb,
                count=count,
                frequency=count / max(1, n_opaque),
                L=float(lch[i, 0]),
                chroma=float(lch[i, 1]),
                hue=float(lch[i, 2]),
                boundary_fraction=on_boundary / max(1, count),
                boundary_share=on_boundary / n_boundary,
            )
        )
    assign_roles(infos)
    return infos


_SMALL_N_ROLES: dict[int, list[ColorRole]] = {
    1: [ColorRole.BASE],
    2: [ColorRole.SHADOW, ColorRole.BASE],
    3: [ColorRole.SHADOW, ColorRole.BASE, ColorRole.LIGHT],
    4: [ColorRole.DEEP_SHADOW, ColorRole.SHADOW, ColorRole.BASE, ColorRole.LIGHT],
}


def assign_roles(infos: list[ColorInfo]) -> None:
    """Role assignment from lightness plus boundary adjacency (dark != outline by itself)."""
    if not infos:
        return
    by_L = sorted(infos, key=lambda c: c.L)
    # Outline: the darkest colour that owns a meaningful share of the exterior boundary.
    # Gen III sprites also run coloured outlines along lit edges, so "most boundary" is wrong;
    # "darkest with real boundary presence" is what artists treat as the outline colour.
    outline = None
    candidates = [c for c in by_L if c.L < 45 and c.boundary_share >= 0.1]
    if candidates:
        outline = candidates[0]
    elif by_L[0].L < 30 and by_L[0].boundary_share >= 0.05:
        outline = by_L[0]
    if outline is not None:
        outline.role = ColorRole.OUTLINE
    rest = [c for c in by_L if c is not outline]
    # Very dark, low-frequency colours that are not the outline are internal linework.
    for c in list(rest):
        if outline is not None and c.L < 25 and c.frequency < 0.08:
            c.role = ColorRole.INTERNAL_LINE
            rest.remove(c)
    n = len(rest)
    if n == 0:
        return
    if n in _SMALL_N_ROLES:
        for c, role in zip(rest, _SMALL_N_ROLES[n]):
            c.role = role
        return
    buckets = [ColorRole.DEEP_SHADOW, ColorRole.SHADOW, ColorRole.BASE, ColorRole.LIGHT, ColorRole.HIGHLIGHT]
    for i, c in enumerate(rest):
        p = i / (n - 1)
        c.role = buckets[min(4, int(p * 5))]


def strip_outer_antialias(idx: np.ndarray, opaque: np.ndarray, infos: list[ColorInfo]) -> np.ndarray:
    """Remove light single pixels sitting outside the dark outline (Yellow-style manual AA).

    Such a pixel is on the silhouette edge, light, touches a dark pixel, and has at most two
    opaque 4-neighbours (it fills a stair step rather than belonging to a body region).
    Returns the pixels to drop.
    """
    boundary = exterior_boundary(opaque)
    L = np.zeros(idx.shape)
    dark = np.zeros(idx.shape, dtype=bool)
    for i, c in enumerate(infos):
        L[idx == i] = c.L
        if c.L < 35:
            dark[idx == i] = True
    light_edge = boundary & (L > 55)
    n4 = ndimage.convolve(opaque.astype(np.int32), FOUR.astype(np.int32), mode="constant") - opaque.astype(np.int32)
    dark_neigh = ndimage.binary_dilation(dark, structure=EIGHT) & ~dark
    drop = light_edge & dark_neigh & (n4 <= 2)
    return drop


def connected_component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask, structure=EIGHT)
    return int(n)


def checker_dither(idx: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """Find GBC-style checkerboard dithering between two colours.

    Returns (component mask, colour a, colour b) for each dithered patch of >= 4 pixels.
    A pixel is dithered when 3+ of its 4-neighbours share one other colour and 2+ diagonal
    neighbours share its own colour; a relaxed second pass grows patches by one pixel.
    """
    h, w = idx.shape
    p = np.pad(idx, 1, constant_values=-1)
    n4 = np.stack([p[0:h, 1 : w + 1], p[2 : h + 2, 1 : w + 1], p[1 : h + 1, 0:w], p[1 : h + 1, 2 : w + 2]])
    d4 = np.stack([p[0:h, 0:w], p[0:h, 2 : w + 2], p[2 : h + 2, 0:w], p[2 : h + 2, 2 : w + 2]])
    same_diag = (d4 == idx).sum(axis=0)
    n_other = np.zeros((h, w), dtype=np.int32)
    for k in range(4):
        cand = n4[k]
        valid = (cand >= 0) & (cand != idx)
        count = (n4 == cand).sum(axis=0)
        n_other = np.maximum(n_other, np.where(valid, count, 0))
    strict = (idx >= 0) & (n_other >= 3) & (same_diag >= 2)
    relaxed = (idx >= 0) & (n_other >= 2) & (same_diag >= 1)
    grown = strict | (relaxed & ndimage.binary_dilation(strict, structure=FOUR))
    labels, n = ndimage.label(grown, structure=EIGHT)
    out: list[tuple[np.ndarray, int, int]] = []
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() < 4:
            continue
        vals, counts = np.unique(idx[comp], return_counts=True)
        if len(vals) < 2:
            continue
        order = np.argsort(-counts)
        out.append((comp, int(vals[order[0]]), int(vals[order[1]])))
    return out


def shape_descriptors(mask: np.ndarray, rgba: np.ndarray | None = None) -> dict[str, float]:
    """Simple silhouette descriptors used for auto-reference ranking."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return {}
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    area = float(mask.sum())
    boundary = exterior_boundary(mask)
    perimeter = float(boundary.sum())
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    flipped = crop[:, ::-1]
    symmetry = float((crop & flipped).sum() / max(1, (crop | flipped).sum()))
    vertical_com = float((ys.mean() - ys.min()) / max(1, h - 1))
    # Protrusions: convex-hull-free proxy via difference between area and its 3px closing.
    closed = ndimage.binary_closing(np.pad(crop, 4), structure=np.ones((7, 7), bool))[4:-4, 4:-4]
    protrusion = float((closed & ~crop).sum() / max(1.0, area))
    desc = {
        "aspect": float(w / max(1, h)),
        "area_ratio": float(area / (w * h)),
        "compactness": float(4 * np.pi * area / max(1.0, perimeter**2)),
        "complexity": float(perimeter / np.sqrt(max(1.0, area))),
        "protrusion": protrusion,
        "vertical_com": vertical_com,
        "symmetry": symmetry,
        "edge_density": float(perimeter / area),
        "longest_axis": float(max(h, w)),
    }
    if rgba is not None:
        lch = rgb_to_lch(rgba[mask][:, :3])
        chromatic = lch[:, 1] > 15
        if chromatic.any():
            hs = np.radians(lch[chromatic, 2])
            desc["hue_x"] = float(np.cos(hs).mean())
            desc["hue_y"] = float(np.sin(hs).mean())
        else:
            desc["hue_x"] = 0.0
            desc["hue_y"] = 0.0
    return desc
