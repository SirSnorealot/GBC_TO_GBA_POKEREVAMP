"""Conversion report (JSON) written next to every output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gbc_to_gba_pokerevamp.models import RGB


@dataclass
class ConversionReport:
    input: str
    output: str
    kind: str
    style: str
    reference_files: list[str] = field(default_factory=list)
    source_dimensions: tuple[int, int] = (0, 0)
    source_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_dimensions: tuple[int, int] = (0, 0)
    target_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_opaque_colors: int = 0
    target_opaque_colors: int = 0
    source_palette: list[RGB] = field(default_factory=list)
    target_palette: list[RGB] = field(default_factory=list)
    source_roles: dict[str, str] = field(default_factory=dict)
    families: list[dict[str, Any]] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)
    reference_style: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
