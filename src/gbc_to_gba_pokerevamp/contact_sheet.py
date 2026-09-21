"""Contact sheet: Crystal source | revamp output | official Gen III reference, per row."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from gbc_to_gba_pokerevamp.sprite_io import load_sprite

CELL = 64
SCALE = 3
PAD = 6
LABEL_H = 12


def make_contact_sheet(rows: list[tuple[Path, Path, Path | None]], output: Path, scale: int = SCALE) -> Image.Image:
    cols = ["source", "revamp", "reference"]
    cell = CELL * scale
    w = len(cols) * (cell + PAD) + PAD
    h = len(rows) * (cell + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (w, h), (90, 90, 90))
    draw = ImageDraw.Draw(sheet)
    for r, (src, out, ref) in enumerate(rows):
        y = PAD + r * (cell + LABEL_H + PAD)
        draw.text((PAD, y), f"{src.stem}", fill=(255, 255, 255))
        for c, path in enumerate((src, out, ref)):
            x = PAD + c * (cell + PAD)
            draw.rectangle([x, y + LABEL_H, x + cell - 1, y + LABEL_H + cell - 1], fill=(160, 160, 160))
            if path is None or not path.exists():
                draw.text((x + 4, y + LABEL_H + 4), "n/a", fill=(30, 30, 30))
                continue
            img = load_sprite(path).image
            big = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
            ox = x + max(0, (cell - big.width) // 2)
            oy = y + LABEL_H + max(0, (cell - big.height) // 2)
            sheet.paste(big, (ox, oy), big)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return sheet
