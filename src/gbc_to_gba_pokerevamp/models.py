"""Core data models shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

RGB = tuple[int, int, int]
Kind = Literal["pokemon", "trainer", "unknown"]


class ColorRole(IntEnum):
    BACKGROUND = 0
    OUTLINE = 1
    INTERNAL_LINE = 2
    DEEP_SHADOW = 3
    SHADOW = 4
    BASE = 5
    LIGHT = 6
    HIGHLIGHT = 7
    ACCENT = 8


# Diagnostic colours for role maps only; these never reach the final output.
ROLE_DEBUG_COLORS: dict[ColorRole, RGB] = {
    ColorRole.BACKGROUND: (40, 40, 40),
    ColorRole.OUTLINE: (0, 0, 0),
    ColorRole.INTERNAL_LINE: (120, 40, 140),
    ColorRole.DEEP_SHADOW: (40, 40, 160),
    ColorRole.SHADOW: (60, 110, 220),
    ColorRole.BASE: (80, 200, 80),
    ColorRole.LIGHT: (240, 220, 80),
    ColorRole.HIGHLIGHT: (255, 255, 255),
    ColorRole.ACCENT: (240, 60, 60),
}


@dataclass
class ColorInfo:
    rgb: RGB
    count: int
    frequency: float
    L: float
    chroma: float
    hue: float
    boundary_fraction: float  # fraction of this colour's pixels on the exterior boundary
    boundary_share: float  # fraction of all exterior boundary pixels that have this colour
    role: ColorRole = ColorRole.BASE


@dataclass
class SpriteImage:
    image: Image.Image
    rgba: np.ndarray  # (H, W, 4) uint8
    opaque_mask: np.ndarray  # (H, W) bool
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    palette: list[RGB]
    background_color: RGB | None
    kind: Kind = "unknown"
    source_path: Path | None = None
    original_mode: str = "RGBA"
    warnings: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.shape[1], self.rgba.shape[0]

    @property
    def content_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bbox
        return x1 - x0, y1 - y0


@dataclass
class ReferenceStyle:
    """Relative shading behaviour learned from one or more Gen III references."""

    canvas_size: int = 64
    max_colors: int = 16
    palette: list[RGB] = field(default_factory=list)
    outline_colors: list[RGB] = field(default_factory=list)
    highlight_colors: list[RGB] = field(default_factory=list)
    shadow_colors: list[RGB] = field(default_factory=list)
    # Relative shade deltas in LCh space (shadow/light relative to base of the same hue family).
    shadow_dL: float = 14.0
    deep_dL: float = 26.0
    light_dL: float = 12.0
    highlight_dL: float = 22.0
    shadow_chroma_ratio: float = 1.05
    light_chroma_ratio: float = 0.85
    shadow_hue_shift: float = 8.0  # degrees toward the cool hue
    light_hue_shift: float = 6.0  # degrees toward the warm hue
    outline_L: float = 12.0
    outline_chroma: float = 8.0
    # Share of silhouette-outline pixels that are pure black vs a lighter body-colour shade.
    outline_black_fraction: float = 0.45
    lit_outline_fraction: float = 0.15
    mean_chroma: float = 40.0
    mean_L: float = 55.0
    # Share of family pixels at each relative level (measured on references, used as targets).
    shadow_fraction: float = 0.30  # level < 0
    deep_fraction: float = 0.08  # level <= -2
    light_fraction: float = 0.18  # level > 0
    highlight_fraction: float = 0.04  # level >= 2
    outline_fraction: float = 0.2
    occupancy: float = 52.0  # longest content axis in pixels
    opaque_color_count: int = 12
    # Reference hue families: {base_lch, levels: {"-2".."2": rgb}, pixels, achromatic, weight}
    families: list[dict] = field(default_factory=list)
    same_subject: bool = False  # the top reference is the same species/character as the source
    adopt_colors: bool = False  # pull source colours toward the reference (same species or user-chosen)
    # Coarse map (GRID x GRID) of which family index covers each part of the reference's bbox.
    family_grid: np.ndarray | None = None
    reference_paths: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("reference_paths", "family_grid")}
        d["reference_paths"] = [str(p) for p in self.reference_paths]
        return d


@dataclass
class Ramp:
    """Shade ramp derived for one source local colour."""

    source: RGB
    deep: RGB
    shadow: RGB
    base: RGB
    light: RGB
    highlight: RGB
    line: RGB

    def at(self, level: int) -> RGB:
        if level <= -2:
            return self.deep
        if level == -1:
            return self.shadow
        if level == 0:
            return self.base
        if level == 1:
            return self.light
        return self.highlight
