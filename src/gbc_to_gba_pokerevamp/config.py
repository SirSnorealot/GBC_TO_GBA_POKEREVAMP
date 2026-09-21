"""Revamp configuration model and JSON config-file loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Style = Literal["frlg", "emerald", "gen3-mixed"]
PaletteMode = Literal["source-expanded", "reference-guided", "reference-palette"]
GeometryMode = Literal["none", "integer", "pixel-aware"]
ReferenceMode = Literal["style", "palette"]


class RevampConfig(BaseModel):
    kind: Literal["pokemon", "trainer", "unknown"] = "unknown"
    style: Style = "frlg"
    canvas_size: int = 64
    max_colors: int = 16  # including transparency -> 15 opaque
    preserve_pose: bool = True
    # "none" keeps the source pixel size and only centers it on the canvas (Gen III sprites are
    # redrawn larger, but enlarging pixel art by a fraction only produces uneven blocks).
    geometry_mode: GeometryMode = "none"
    palette_mode: PaletteMode = "reference-guided"
    reference_mode: ReferenceMode = "style"
    auto_reference: bool = True
    reference_count: int = 3
    shading_strength: float = Field(0.7, ge=0.0, le=1.0)
    outline_strength: float = Field(0.75, ge=0.0, le=1.0)
    hue_shift_strength: float = Field(0.6, ge=0.0, le=1.0)
    highlight_strength: float = Field(0.45, ge=0.0, le=1.0)
    cleanup_strength: float = Field(0.5, ge=0.0, le=1.0)
    reference_weight: float = Field(0.65, ge=0.0, le=1.0)
    source_weight: float = Field(0.35, ge=0.0, le=1.0)
    transparent_output: bool = True
    indexed_output: bool = False
    light_x: float = -1.0
    light_y: float = -1.0
    # Target longest-axis occupancy range on the canvas (pixels).
    occupancy_min: int = 46
    occupancy_max: int = 58
    anchor_x: int | None = None
    anchor_y: int | None = None
    debug: bool = False
    report: bool = True
    compare: bool = True  # write a source | revamp | reference sheet next to every result
    seed: int = 0
    # Effect toggles (the GUI exposes these; all on by default).
    recolor: bool = True  # same-species positional recolor
    force_same_subject: bool = False  # treat an explicitly chosen reference as the same Pokémon
    rebuild_outline: bool = True
    shade: bool = True
    enforce_palette: bool = True
    # Manual color mapping: source "r,g,b" -> target "r,g,b" (a reference/any color),
    # or one of "auto", "keep", "outline", "transparent".
    color_map: dict[str, str] = Field(default_factory=dict)

    @property
    def max_opaque_colors(self) -> int:
        return max(1, self.max_colors - 1) if self.transparent_output else self.max_colors

    @classmethod
    def from_file(cls, path: Path, overrides: dict[str, Any] | None = None) -> "RevampConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a JSON object")
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def with_overrides(self, overrides: dict[str, Any]) -> "RevampConfig":
        return self.model_copy(update={k: v for k, v in overrides.items() if v is not None})
