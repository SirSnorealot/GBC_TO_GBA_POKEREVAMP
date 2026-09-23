"""Image loading/saving with strict pixel-art semantics (no resampling, hard alpha)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from gbc_to_gba_pokerevamp.analyze import detect_background
from gbc_to_gba_pokerevamp.models import RGB, Kind, SpriteImage


class SpriteLoadError(Exception):
    pass


def open_image(path: Path) -> Image.Image:
    if not path.exists():
        raise SpriteLoadError(f"Input file not found: {path}")
    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise SpriteLoadError(f"Unsupported or corrupt image: {path} ({exc})") from exc
    return img


def image_to_rgba(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (rgba array, palette index array or None). Never resamples."""
    indices: np.ndarray | None = None
    if img.mode == "P":
        indices = np.array(img, dtype=np.int32)
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    return rgba, indices


def load_sprite(path: Path, kind: Kind = "unknown") -> SpriteImage:
    img = open_image(path)
    original_mode = img.mode
    rgba, indices = image_to_rgba(img)
    opaque, bg_color, warnings = detect_background(rgba, indices)
    if not opaque.any():
        raise SpriteLoadError(f"No visible subject found in {path} (everything was classified as background)")
    ys, xs = np.nonzero(opaque)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    colors = np.unique(rgba[opaque][:, :3], axis=0)
    palette: list[RGB] = [tuple(int(v) for v in c) for c in colors]  # type: ignore[misc]
    # Hard alpha: everything not background is fully opaque.
    rgba = rgba.copy()
    rgba[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    rgba[~opaque, :3] = 0
    return SpriteImage(
        image=Image.fromarray(rgba, "RGBA"),
        rgba=rgba,
        opaque_mask=opaque,
        bbox=bbox,
        palette=palette,
        background_color=bg_color,
        kind=kind,
        source_path=path,
        original_mode=original_mode,
        warnings=warnings,
    )


def save_rgba_png(rgba: np.ndarray, path: Path) -> None:
    """Save an (H, W, 4) uint8 array as PNG with hard 0/255 alpha."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    alpha = rgba[..., 3]
    if not np.all((alpha == 0) | (alpha == 255)):
        raise ValueError("Refusing to save image with partially transparent pixels")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgba, "RGBA").save(path, format="PNG", optimize=False)
    except OSError as exc:
        raise SpriteLoadError(f"Output path is not writable: {path} ({exc})") from exc


def save_indexed_png(rgba: np.ndarray, path: Path, max_entries: int = 16) -> None:
    """Save as a paletted PNG. Index 0 is transparent (GBA convention)."""
    opaque = rgba[..., 3] > 0
    colors = np.unique(rgba[opaque][:, :3], axis=0)
    if len(colors) > max_entries - 1:
        raise ValueError(f"Image has {len(colors)} opaque colors; indexed export allows {max_entries - 1}")
    lut = {tuple(int(v) for v in c): i + 1 for i, c in enumerate(colors)}
    h, w = opaque.shape
    idx = np.zeros((h, w), dtype=np.uint8)
    flat = rgba[..., :3].reshape(-1, 3)
    out = idx.reshape(-1)
    for i, px in enumerate(flat):
        if opaque.reshape(-1)[i]:
            out[i] = lut[(int(px[0]), int(px[1]), int(px[2]))]
    pal = [0, 0, 0]
    for c in colors:
        pal.extend(int(v) for v in c)
    pal.extend([0] * (768 - len(pal)))
    img = Image.fromarray(idx, "P")
    img.putpalette(pal)
    img.info["transparency"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", transparency=0, optimize=False)


def scale_nearest(img: Image.Image, factor: int) -> Image.Image:
    """Integer nearest-neighbour magnification for previews."""
    if factor <= 1:
        return img.copy()
    return img.resize((img.width * factor, img.height * factor), Image.Resampling.NEAREST)


def find_frame_sheet(path: Path) -> Path | None:
    """Locate the frame sheet belonging to a sprite, if any.

    Accepts either the single-frame file (looks for `<stem>_frames.png`) or a sheet itself
    (a PNG whose height is a multiple of its width).
    """
    stem = path.stem
    for suffix in ("_shiny",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    sheet = path.with_name(stem + "_frames.png")
    if sheet.exists():
        return sheet
    img = open_image(path)
    if img.height > img.width and img.height % img.width == 0:
        return path
    return None


def load_frames(sheet: Path, kind: Kind = "unknown", frame_size: int | None = None) -> list[SpriteImage]:
    """Split a vertical frame sheet into one SpriteImage per frame (all share the sheet's size)."""
    img = open_image(sheet)
    fs = frame_size or img.width
    n = max(1, img.height // fs)
    frames: list[SpriteImage] = []
    for i in range(n):
        crop = img.crop((0, i * fs, img.width, (i + 1) * fs))
        rgba, indices = image_to_rgba(crop)
        opaque, bg_color, warnings = detect_background(rgba, indices)
        if not opaque.any():
            continue
        ys, xs = np.nonzero(opaque)
        rgba = rgba.copy()
        rgba[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
        rgba[~opaque, :3] = 0
        colors = np.unique(rgba[opaque][:, :3], axis=0)
        frames.append(SpriteImage(
            image=Image.fromarray(rgba, "RGBA"), rgba=rgba, opaque_mask=opaque,
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            palette=[tuple(int(v) for v in c) for c in colors],  # type: ignore[misc]
            background_color=bg_color, kind=kind, source_path=sheet, original_mode=img.mode, warnings=warnings,
        ))
    return frames
