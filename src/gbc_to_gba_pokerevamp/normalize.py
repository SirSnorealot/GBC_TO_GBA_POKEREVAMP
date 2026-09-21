"""Geometry normalization onto the GBA battle-sprite canvas (nearest-neighbour / discrete only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.models import ColorRole


@dataclass
class NormalizedSprite:
    idx: np.ndarray  # (canvas, canvas) int32 color indices, -1 = transparent
    mask: np.ndarray  # (canvas, canvas) bool
    scale: float
    mode_used: str
    offset: tuple[int, int]
    content_size: tuple[int, int]
    warnings: list[str]


def _nn_indices(n_src: int, n_dst: int) -> np.ndarray:
    """Source index for each destination pixel using center sampling; purely integer output."""
    return np.minimum(((np.arange(n_dst) + 0.5) * n_src / n_dst).astype(np.int64), n_src - 1)


def nearest_scale_indices(idx: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    ys = _nn_indices(idx.shape[0], new_h)
    xs = _nn_indices(idx.shape[1], new_w)
    return idx[ys][:, xs]


def _duplicated_positions(n_src: int, n_dst: int) -> np.ndarray:
    """Destination positions that repeat the previous destination's source pixel."""
    src = _nn_indices(n_src, n_dst)
    dup = np.zeros(n_dst, dtype=bool)
    dup[1:] = src[1:] == src[:-1]
    return dup


def _is_simple_point(is_line: np.ndarray, r: int, c: int) -> bool:
    """Zhang-Suen deletability: removing this line pixel does not break 8-connectivity."""
    h, w = is_line.shape

    def at(y: int, x: int) -> int:
        return int(is_line[y, x]) if 0 <= y < h and 0 <= x < w else 0

    ring = [at(r - 1, c), at(r - 1, c + 1), at(r, c + 1), at(r + 1, c + 1), at(r + 1, c), at(r + 1, c - 1), at(r, c - 1), at(r - 1, c - 1)]
    b = sum(ring)
    a = sum(1 for i in range(8) if ring[i] == 0 and ring[(i + 1) % 8] == 1)
    return 2 <= b <= 6 and a == 1


def thin_duplicated_lines(scaled: np.ndarray, src_idx: np.ndarray, line_colors: set[int]) -> np.ndarray:
    """Undo 1px->2px thickening of dark lines caused by non-integer nearest-neighbour scaling.

    Only pixels in duplicated rows/columns are touched, only when the duplicate sits next to
    its origin with fill on the far side, and only when removing it keeps the line connected.
    """
    if not line_colors:
        return scaled
    out = scaled.copy()
    h, w = out.shape
    lines = list(line_colors)

    def fillable(v: int) -> bool:
        return v >= 0 and v not in line_colors

    is_line = np.isin(out, lines)
    for c in np.nonzero(_duplicated_positions(src_idx.shape[1], w))[0]:
        if c < 2 or c + 1 >= w:
            continue
        for r in np.nonzero(is_line[:, c] & is_line[:, c - 1])[0]:
            if fillable(out[r, c + 1]) and not is_line[r, c + 1]:
                target, fill = (r, c), out[r, c + 1]
            elif fillable(out[r, c - 2]) and not is_line[r, c - 2]:
                target, fill = (r, c - 1), out[r, c - 2]
            else:
                continue
            if _is_simple_point(is_line, *target):
                out[target] = fill
                is_line[target] = False
    for r in np.nonzero(_duplicated_positions(src_idx.shape[0], h))[0]:
        if r < 2 or r + 1 >= h:
            continue
        for c in np.nonzero(is_line[r, :] & is_line[r - 1, :])[0]:
            if fillable(out[r + 1, c]) and not is_line[r + 1, c]:
                target, fill = (r, c), out[r + 1, c]
            elif fillable(out[r - 2, c]) and not is_line[r - 2, c]:
                target, fill = (r - 1, c), out[r - 2, c]
            else:
                continue
            if _is_simple_point(is_line, *target):
                out[target] = fill
                is_line[target] = False
    return out


def choose_scale(content_w: int, content_h: int, config: RevampConfig, target_occupancy: float | None) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    longest = max(content_w, content_h)
    canvas = config.canvas_size
    occ_max = config.occupancy_max if config.kind != "trainer" else min(canvas - 2, config.occupancy_max + 4)
    target = target_occupancy if target_occupancy else (config.occupancy_min + occ_max) / 2
    target = float(np.clip(target, config.occupancy_min, occ_max))
    mode = config.geometry_mode

    if longest > canvas:
        warnings.append(f"subject ({longest}px) is larger than the {canvas}px canvas; downscaling with nearest-neighbour")
        return canvas / longest, "downscale", warnings
    if mode == "none":
        return 1.0, "none", warnings
    if mode == "integer":
        factor = max(1, int(target // longest))
        while longest * factor > canvas:
            factor -= 1
        return float(max(1, factor)), "integer", warnings
    # pixel-aware: modest fractional enlargement, thin doubled lines afterwards.
    factor = target / longest
    factor = min(factor, canvas / longest)
    if factor < 1.12:
        return 1.0, "none", warnings
    if factor > 1.6:
        # Large fractional factors produce very uneven blocks; prefer a clean integer.
        int_factor = int(factor)
        if int_factor >= 2 and longest * int_factor <= canvas:
            return float(int_factor), "integer", warnings
        warnings.append("pixel-aware enlargement factor is large; expect uneven pixel blocks")
    return float(factor), "pixel-aware", warnings


def normalize_geometry(
    idx: np.ndarray,
    bbox: tuple[int, int, int, int],
    roles: dict[int, ColorRole],
    config: RevampConfig,
    target_occupancy: float | None = None,
) -> NormalizedSprite:
    x0, y0, x1, y1 = bbox
    crop = idx[y0:y1, x0:x1]
    ch, cw = crop.shape
    scale, mode_used, warnings = choose_scale(cw, ch, config, target_occupancy)
    canvas = config.canvas_size

    if scale != 1.0:
        new_w = max(1, min(canvas, int(round(cw * scale))))
        new_h = max(1, min(canvas, int(round(ch * scale))))
        scaled = nearest_scale_indices(crop, new_w, new_h)
        if mode_used == "pixel-aware":
            line_colors = {i for i, r in roles.items() if r in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE)}
            scaled = thin_duplicated_lines(scaled, crop, line_colors)
    else:
        scaled = crop
    sh, sw = scaled.shape
    if sh > canvas or sw > canvas:
        scaled = scaled[:canvas, :canvas]
        sh, sw = scaled.shape

    ox = (canvas - sw) // 2 if config.anchor_x is None else int(np.clip(config.anchor_x, 0, canvas - sw))
    oy = (canvas - sh) // 2 if config.anchor_y is None else int(np.clip(config.anchor_y, 0, canvas - sh))
    out = np.full((canvas, canvas), -1, dtype=np.int32)
    out[oy : oy + sh, ox : ox + sw] = scaled
    return NormalizedSprite(
        idx=out,
        mask=out >= 0,
        scale=scale,
        mode_used=mode_used,
        offset=(ox, oy),
        content_size=(sw, sh),
        warnings=warnings,
    )
