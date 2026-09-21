"""Discrete geometry helpers: lighting orientation, morphology, and pixel-cluster cleanup."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.analyze import EIGHT, FOUR


def lighting_dot(mask: np.ndarray, light: tuple[float, float], smooth: int = 5) -> np.ndarray:
    """Per-pixel cosine between the outward surface normal and the light direction.

    The normal is the negative gradient of a lightly smoothed inside-distance field, so it
    describes coarse silhouette orientation rather than single-pixel jaggies. Smoothing is
    used only to derive orientation; it never touches output pixels.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    dist = ndimage.distance_transform_edt(mask).astype(np.float64)
    if smooth > 1:
        dist = ndimage.uniform_filter(dist, size=smooth, mode="constant")
    gy, gx = np.gradient(dist)
    nx, ny = -gx, -gy
    norm = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / norm, ny / norm
    lx, ly = light
    ln = np.hypot(lx, ly) + 1e-9
    dot = nx * (lx / ln) + ny * (ly / ln)
    dot[~mask] = 0.0
    return dot


def inside_distance(mask: np.ndarray) -> np.ndarray:
    """Chessboard-free Euclidean distance to the nearest pixel outside the mask."""
    return ndimage.distance_transform_edt(mask)


def illumination(region: np.ndarray, mask: np.ndarray, light: tuple[float, float]) -> np.ndarray:
    """Per-pixel brightness estimate treating the filled region as a rounded form.

    The inside-distance field is shaped into a dome whose gradient gives a surface normal;
    that is lit by the directional light, plus a top-is-lit vertical term. Smoothing only
    derives orientation; it never touches output pixels.
    """
    filled = ndimage.binary_fill_holes(region) & mask
    if not filled.any():
        return np.zeros(mask.shape, dtype=np.float64)
    dist = ndimage.distance_transform_edt(filled).astype(np.float64)
    r = max(1.0, float(dist.max()))
    height = np.sqrt(np.clip(1.0 - (1.0 - dist / r) ** 2, 0.0, 1.0))
    smooth = ndimage.gaussian_filter(height, sigma=1.5, mode="constant")
    gy, gx = np.gradient(smooth)
    lx, ly = light
    ln = np.hypot(lx, ly) + 1e-9
    lx, ly = lx / ln, ly / ln
    k = 0.12  # how much the flat centre of the form reads as lit
    nz = np.sqrt(gx * gx + gy * gy + k * k)
    lit = (-gx / nz) * lx + (-gy / nz) * ly + (k / nz) * 0.6
    ys = np.nonzero(filled)[0]
    rel_y = np.zeros_like(lit)
    rel_y[filled] = (ys - ys.min()) / max(1, ys.max() - ys.min())
    lit -= 0.35 * (rel_y - 0.5)
    lit = ndimage.gaussian_filter(lit, sigma=1.0, mode="nearest")
    lit[~filled] = 0.0
    return lit


def dilate(mask: np.ndarray, iterations: int = 1, connectivity: int = 4) -> np.ndarray:
    return ndimage.binary_dilation(mask, structure=FOUR if connectivity == 4 else EIGHT, iterations=iterations)


def erode(mask: np.ndarray, iterations: int = 1, connectivity: int = 4) -> np.ndarray:
    return ndimage.binary_erosion(mask, structure=FOUR if connectivity == 4 else EIGHT, iterations=iterations, border_value=0)


def nearest_label(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For every pixel, the label of the nearest `valid` pixel."""
    if not valid.any():
        return labels.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return labels[iy, ix]


def small_components(mask: np.ndarray, max_area: int, connectivity: int = 8) -> np.ndarray:
    """Mask of connected components whose area is <= max_area."""
    labels, n = ndimage.label(mask, structure=EIGHT if connectivity == 8 else FOUR)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    small = np.nonzero(sizes <= max_area)[0] + 1
    return np.isin(labels, small)


def _neighbors8(arr: np.ndarray, fill) -> list[np.ndarray]:
    p = np.pad(arr, 1, constant_values=fill)
    h, w = arr.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out.append(p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
    return out


def count_isolated_pixels(rgba: np.ndarray) -> int:
    opaque = rgba[..., 3] > 0
    key = _color_key(rgba)
    same = np.zeros(opaque.shape, dtype=bool)
    for n in _neighbors8(key, -1):
        same |= n == key
    return int((opaque & ~same).sum())


def _color_key(rgba: np.ndarray) -> np.ndarray:
    key = rgba[..., 0].astype(np.int64) * 65536 + rgba[..., 1].astype(np.int64) * 256 + rgba[..., 2].astype(np.int64)
    return np.where(rgba[..., 3] > 0, key, -1)


def remove_isolated_pixels(rgba: np.ndarray, protected: np.ndarray, keep: np.ndarray, strength: float) -> tuple[np.ndarray, int]:
    """Replace opaque pixels that share their colour with no 8-neighbour by their dominant neighbour.

    `protected` pixels (accents) and `keep` pixels (exterior outline) are never modified.
    At strength >= 0.75, two-pixel islands are treated the same way.
    """
    if strength <= 0:
        return rgba, 0
    out = rgba.copy()
    changed = 0
    max_area = 2 if strength >= 0.75 else 1
    for _ in range(2):
        key = _color_key(out)
        opaque = key >= 0
        labels = np.zeros(key.shape, dtype=np.int64)
        # Label same-colour components by processing each colour separately.
        next_label = 1
        for k in np.unique(key[opaque]):
            lab, n = ndimage.label(key == k, structure=EIGHT)
            labels[lab > 0] = lab[lab > 0] + next_label
            next_label += n + 1
        sizes = np.bincount(labels.reshape(-1))
        island = opaque & (sizes[labels] <= max_area) & ~protected & ~keep
        if not island.any():
            break
        neigh = _neighbors8(key, -1)
        ys, xs = np.nonzero(island)
        for y, x in zip(ys, xs):
            vals = [n[y, x] for n in neigh if n[y, x] >= 0 and n[y, x] != key[y, x]]
            if not vals:
                continue
            best = max(set(vals), key=vals.count)
            out[y, x, 0] = (best >> 16) & 255
            out[y, x, 1] = (best >> 8) & 255
            out[y, x, 2] = best & 255
            changed += 1
    return out, changed
