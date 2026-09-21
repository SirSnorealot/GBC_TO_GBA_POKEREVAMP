"""Compose final pixels from masks/roles, plus diagnostic and comparison renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from gbc_to_gba_pokerevamp.geometry import nearest_label
from gbc_to_gba_pokerevamp.models import RGB, ROLE_DEBUG_COLORS, ColorRole
from gbc_to_gba_pokerevamp.outline import LINE_BASE, LINE_BLACK, LINE_EXTREME, LINE_HIGHLIGHT
from gbc_to_gba_pokerevamp.palette import Family


@dataclass
class RenderState:
    """Everything the renderer needs; stages only ever edit these separate layers."""

    idx: np.ndarray  # normalised source colour index, -1 transparent
    mask: np.ndarray
    family_map: np.ndarray  # >=0 family id, -2 source dark/line pixel, -1 transparent
    level: np.ndarray  # int8 ramp level per pixel
    line: np.ndarray  # LINE_* codes
    protected: np.ndarray
    families: list[Family]
    outline_rgb: RGB
    source_palette: list[RGB]
    source_roles: dict[int, ColorRole]
    line_family: np.ndarray | None = field(default=None)

    def family_lookup(self) -> np.ndarray:
        """Family that colours each line pixel; set by the outline stage, else nearest body pixel."""
        if self.line_family is None:
            valid = (self.family_map >= 0) & ~self.protected
            if not valid.any():
                valid = self.family_map >= 0
            self.line_family = nearest_label(self.family_map, valid) if valid.any() else self.family_map.copy()
        return self.line_family


def render(state: RenderState) -> np.ndarray:
    h, w = state.mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ramps = {f.id: f.ramp for f in state.families if f.ramp is not None}
    near = state.family_lookup()
    ys, xs = np.nonzero(state.mask)
    for y, x in zip(ys, xs):
        fam = int(state.family_map[y, x])
        code = int(state.line[y, x])
        owner = fam if (fam >= 0 and code == 0) else int(near[y, x])
        ramp = ramps.get(owner)
        if code == LINE_BLACK or (code and ramp is None) or (fam < 0 and ramp is None):
            rgb = state.outline_rgb
        elif code == LINE_BASE:
            rgb = ramp.line
        elif code == LINE_HIGHLIGHT:
            rgb = ramp.deep
        elif code == LINE_EXTREME:
            rgb = ramp.shadow
        elif ramp is not None:
            rgb = ramp.at(int(state.level[y, x]))
        else:
            rgb = state.source_palette[int(state.idx[y, x])]
        out[y, x, :3] = rgb
        out[y, x, 3] = 255
    return out


def render_role_map(idx: np.ndarray, roles: dict[int, ColorRole]) -> np.ndarray:
    h, w = idx.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for i, role in roles.items():
        m = idx == i
        out[m, :3] = ROLE_DEBUG_COLORS[role]
        out[m, 3] = 255
    return out


def render_level_map(mask: np.ndarray, level: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Diagnostic view: blue=deep, cyan=shadow, grey=base, yellow=light, white=highlight, black=line."""
    colors = {-2: (40, 40, 160), -1: (60, 110, 220), 0: (128, 128, 128), 1: (240, 220, 80), 2: (255, 255, 255)}
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for lvl, c in colors.items():
        m = mask & (level == lvl) & (line == 0)
        out[m, :3] = c
    out[mask & (line == LINE_BLACK), :3] = (0, 0, 0)
    out[mask & (line == LINE_BASE), :3] = (120, 40, 140)
    out[mask & (line == LINE_HIGHLIGHT), :3] = (200, 120, 40)
    out[mask & (line == LINE_EXTREME), :3] = (255, 180, 80)
    out[mask, 3] = 255
    return out


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[mask] = (255, 255, 255, 255)
    out[~mask] = (0, 0, 0, 255)
    return out


def index_to_rgba(idx: np.ndarray, palette: list[RGB]) -> np.ndarray:
    h, w = idx.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    pal = np.array(palette, dtype=np.uint8) if palette else np.zeros((1, 3), np.uint8)
    m = idx >= 0
    out[m, :3] = pal[idx[m]]
    out[m, 3] = 255
    return out


CHECKER_A = (200, 200, 200)
CHECKER_B = (160, 160, 160)


def _checkerboard(w: int, h: int, cell: int = 8) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    tile = ((yy // cell + xx // cell) % 2).astype(bool)
    arr = np.where(tile[..., None], np.array(CHECKER_A, np.uint8), np.array(CHECKER_B, np.uint8))
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def compare_sheet(images: list[Image.Image], labels: list[str], scale: int = 4, pad: int = 8) -> Image.Image:
    """Side-by-side sheet: each image at 1x and at an integer magnification, nearest-neighbour only."""
    scale = max(1, int(scale))
    label_h = 14
    cells = []
    for img in images:
        rgba = img.convert("RGBA")
        big = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
        cells.append((rgba, big))
    col_w = max(c[1].width for c in cells) + pad * 2
    row_h = label_h + max(c[0].height for c in cells) + pad + max(c[1].height for c in cells) + pad * 2
    sheet = _checkerboard(col_w * len(cells), row_h)
    draw = ImageDraw.Draw(sheet)
    for i, ((small, big), label) in enumerate(zip(cells, labels)):
        x = i * col_w + pad
        draw.rectangle([i * col_w, 0, i * col_w + col_w, label_h], fill=(30, 30, 30))
        draw.text((x, 1), label[:40], fill=(255, 255, 255))
        y = label_h + pad
        sheet.paste(small, (x, y), small)
        y += small.height + pad
        sheet.paste(big, (x, y), big)
    return sheet
