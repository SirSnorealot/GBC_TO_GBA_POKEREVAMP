"""Target palette construction: hue families, shade ramps, and palette-limit enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gbc_to_gba_pokerevamp.colorspace import hue_delta, lch_to_rgb, rgb_to_lab, rgb_to_lch, shift_hue_toward
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.models import RGB, ColorInfo, ColorRole, Ramp, ReferenceStyle

COOL_HUE = 275.0  # shadows drift toward blue/violet
WARM_HUE = 85.0  # lights drift toward yellow
ACHROMATIC_CHROMA = 12.0
FAMILY_HUE_TOLERANCE = 42.0
REFERENCE_MATCH_HUE = 35.0  # max hue distance for adopting a reference family's colours


@dataclass
class Family:
    id: int
    colors: list[ColorInfo]
    base: ColorInfo
    levels: dict[RGB, int] = field(default_factory=dict)  # source colour -> ramp level
    achromatic: bool = False
    ramp: Ramp | None = None

    @property
    def pixel_count(self) -> int:
        return sum(c.count for c in self.colors)


def group_families(infos: list[ColorInfo], kind: str = "unknown", step_levels: bool = False) -> list[Family]:
    """Cluster non-line source colours into hue families; each family becomes one shade ramp.

    `step_levels=True` (used for 16-colour references) assigns levels by lightness distance
    from the base so two near-identical shades share a level instead of eating the light and
    highlight slots; GBC sources with 1-2 colours per family use plain rank order.
    """
    tolerance = FAMILY_HUE_TOLERANCE if kind != "trainer" else 28.0
    locals_ = [c for c in infos if c.role not in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE)]
    locals_.sort(key=lambda c: -c.count)
    families: list[Family] = []
    for c in locals_:
        achro = c.chroma < ACHROMATIC_CHROMA
        joined = False
        for f in families:
            if achro != f.achromatic:
                continue
            if achro:
                # Whites/greys join a single grey family only when both are light; dark greys stay separate.
                if (c.L > 60) == (f.base.L > 60):
                    f.colors.append(c)
                    joined = True
                    break
                continue
            d = abs((c.hue - f.base.hue + 180.0) % 360.0 - 180.0)
            # GBC shadow colours often sit far around the hue wheel from the body colour
            # (Pikachu's red-orange under yellow), so clearly darker colours get more slack.
            slack = tolerance + 30.0 if c.L < f.base.L - 20.0 else tolerance
            # Hue is unreliable for low-chroma colours (Gen III teal/grey-green shadows).
            slack += 1.5 * max(0.0, 25.0 - min(c.chroma, f.base.chroma))
            if d <= slack:
                f.colors.append(c)
                joined = True
                break
        if not joined:
            families.append(Family(id=len(families), colors=[c], base=c, achromatic=achro))
    for f in families:
        f.base = max(f.colors, key=lambda c: c.count)
        lighter = sorted([c for c in f.colors if c.L > f.base.L], key=lambda c: c.L)
        darker = sorted([c for c in f.colors if c.L < f.base.L], key=lambda c: -c.L)
        f.levels[f.base.rgb] = 0
        if step_levels and len(f.colors) > 2:
            # Levels by lightness distance from the base in steps of a quarter of the family's
            # range, so near-identical shades share a level and the real light/highlight tones
            # land on +1/+2 instead of being used up by a 5-L-apart sibling of the base.
            Ls = [c.L for c in f.colors]
            step = max(6.0, (max(Ls) - min(Ls)) / 4.0)
            for c in f.colors:
                d = (c.L - f.base.L) / step
                f.levels[c.rgb] = int(np.clip(np.sign(d) * np.floor(abs(d) + 0.5), -2, 2))
            continue
        for i, c in enumerate(lighter):
            f.levels[c.rgb] = min(2, i + 1)
        for i, c in enumerate(darker):
            f.levels[c.rgb] = max(-2, -(i + 1))
    return families


def _lch(rgb: RGB) -> np.ndarray:
    return rgb_to_lch(np.array(rgb, dtype=np.uint8))


def _rgb(lch: np.ndarray) -> RGB:
    L, C, h = float(lch[0]), float(lch[1]), float(lch[2])
    out = lch_to_rgb(np.array([np.clip(L, 0, 100), max(0.0, C), h % 360.0]))
    return int(out[0]), int(out[1]), int(out[2])


def _blend_style(ref: ReferenceStyle, config: RevampConfig) -> ReferenceStyle:
    """Blend reference deltas with the neutral defaults according to reference_weight."""
    default = ReferenceStyle()
    w = config.reference_weight if config.palette_mode != "source-expanded" else 0.0
    out = ReferenceStyle()
    for name in (
        "shadow_dL", "deep_dL", "light_dL", "highlight_dL", "shadow_chroma_ratio",
        "light_chroma_ratio", "shadow_hue_shift", "light_hue_shift", "outline_L", "outline_chroma",
        "outline_black_fraction", "lit_outline_fraction", "shadow_fraction", "deep_fraction",
        "light_fraction", "highlight_fraction",
    ):
        setattr(out, name, w * getattr(ref, name) + (1 - w) * getattr(default, name))
    out.outline_colors = list(ref.outline_colors)
    out.palette = list(ref.palette)
    out.families = list(ref.families)
    out.same_subject = ref.same_subject
    out.adopt_colors = ref.adopt_colors
    return out


def _blend_lch(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Interpolate two LCh colours, taking the shortest hue arc."""
    h = (float(a[2]) + w * hue_delta(float(a[2]), float(b[2]))) % 360.0
    return np.array([(1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1], h])


def match_reference_family(family: Family, style: ReferenceStyle, dominant: bool = False) -> dict | None:
    """Find the reference colour family with the same hue role as this source family.

    Only same-hue matches count (yellow->yellow, white->white), so a green character never
    inherits a blue reference's colours; it just falls back to relative shade deltas. The one
    exception is the dominant body colour of a same-species Pokémon reference, which may have
    been re-hued officially (Crystal green Bulbasaur -> FRLG teal).
    """
    best, best_score = None, 0.0
    for rf in style.families:
        # The best-ranked reference always wins; pixel share only breaks ties within one reference.
        rank_score = rf["weight"] * 1000.0
        share = rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"]))
        ref_L, ref_C, ref_h = rf["base_lch"]
        if family.achromatic:
            if family.base.L > 60:
                # Whites may adopt the reference's white/cream, never a saturated colour.
                if not (ref_L >= 72 and ref_C <= 45):
                    continue
            elif style.same_subject and dominant:
                # A black GBC body (Snorlax) may take the official dark body colour.
                if ref_L > 45:
                    continue
            elif not (rf["achromatic"] and ref_L <= 60):
                continue
            score = rank_score + share
        else:
            if rf["achromatic"]:
                continue
            d = abs(hue_delta(family.base.hue, ref_h))
            if style.same_subject and dominant:
                if d > 90.0:
                    continue
                score = rank_score + 2.0 * share * (1.0 - d / 180.0)
            else:
                if d > REFERENCE_MATCH_HUE:
                    continue
                score = rank_score + share * (1.0 - d / (2 * REFERENCE_MATCH_HUE))
        if score > best_score:
            best, best_score = rf, score
    return best


def build_ramp(base_rgb: RGB, family: Family, style: ReferenceStyle, config: RevampConfig, dominant: bool = False) -> Ramp:
    """Derive deep/shadow/base/light/highlight/line from a source colour in LCh space.

    Existing source shades are kept where the family already has them; missing levels are
    synthesised from the (blended) reference deltas. In reference-guided mode every level is
    then pulled toward the same-hue reference family by `reference_weight`, which is what
    makes a Crystal yellow read as FRLG yellow without recolouring unrelated hues.
    """
    base = _lch(base_rgb)
    L, C, h = float(base[0]), float(base[1]), float(base[2])
    s = config.shading_strength * 0.5 + 0.5  # keep some separation even at low strength
    hs = config.hue_shift_strength if not family.achromatic else 0.0
    if config.kind == "trainer":
        hs *= 0.5
    if family.achromatic:
        C = 0.0

    ref_family = match_reference_family(family, style, dominant) if config.palette_mode == "reference-guided" and style.adopt_colors else None
    # A same-species reference is the authoritative palette ("switch out the colours"); a
    # user-chosen reference pulls by reference_weight; auto-picked shape-alikes never recolour.
    w = (1.0 if style.same_subject else config.reference_weight) if ref_family else 0.0
    if config.kind == "trainer" and not family.achromatic:
        w *= 0.5  # skin/hair/clothing hues must stay closer to the source

    def adapt(rgb: RGB, level: int) -> RGB:
        if not ref_family or w <= 0:
            return rgb
        ref_rgb = ref_family["levels"].get(str(level))
        if ref_rgb is None:
            return rgb
        return _rgb(_blend_lch(_lch(rgb), _lch(tuple(ref_rgb)), w))

    def shade(dL: float, cr: float, hue_target: float, hue_deg: float) -> RGB:
        new_h = shift_hue_toward(h, hue_target, hue_deg * hs) if C > 0 else h
        new_C = C * cr if C > 0 else 0.0
        return _rgb(np.array([L + dL, new_C, new_h]))

    by_level: dict[int, RGB] = {}
    for rgb, lvl in family.levels.items():
        by_level.setdefault(lvl, rgb)

    base_out = adapt(base_rgb, 0)
    shadow = by_level.get(-1) or shade(-style.shadow_dL * s, style.shadow_chroma_ratio, COOL_HUE, style.shadow_hue_shift)
    shadow = adapt(shadow, -1)
    sh = _lch(shadow)
    deep = by_level.get(-2)
    if deep is None:
        dl = max(style.deep_dL - style.shadow_dL, 8.0) * s
        new_h = shift_hue_toward(float(sh[2]), COOL_HUE, style.shadow_hue_shift * 0.6 * hs) if sh[1] > 0 else float(sh[2])
        deep = _rgb(np.array([float(sh[0]) - dl, float(sh[1]) * 1.02, new_h]))
    deep = adapt(deep, -2)
    light = by_level.get(1)
    headroom = 100.0 - L
    if light is None:
        light = shade(min(style.light_dL * s, headroom * 0.55), style.light_chroma_ratio, WARM_HUE, style.light_hue_shift)
    light = adapt(light, 1)
    lt = _lch(light)
    highlight = by_level.get(2)
    if highlight is None:
        dl = min(max(style.highlight_dL - style.light_dL, 6.0) * s, max(0.0, 100.0 - float(lt[0])) * 0.6)
        highlight = _rgb(np.array([float(lt[0]) + dl, float(lt[1]) * 0.8, float(lt[2])]))
    highlight = adapt(highlight, 2)
    dp = _lch(deep)
    # Outline base: a dark shade of the local colour that still reads as a line against every
    # body shade. Gen III keeps it fairly saturated (Charizard's outline is dark orange, not black).
    line_L = float(np.clip(min(float(dp[0]) - 12.0, 34.0), 14.0, 34.0))
    line = _rgb(np.array([line_L, min(float(dp[1]) * 1.1, 60.0), float(dp[2])]))
    if ref_family and ref_family.get("line") and w > 0:
        line = _rgb(_blend_lch(_lch(line), _lch(tuple(ref_family["line"])), w))
    return Ramp(source=base_rgb, deep=deep, shadow=shadow, base=base_out, light=light, highlight=highlight, line=line)


def outline_color(infos: list[ColorInfo], style: ReferenceStyle, config: RevampConfig | None = None) -> RGB:
    src = next((c for c in infos if c.role == ColorRole.OUTLINE), None)
    if src is None:
        src = min(infos, key=lambda c: c.L)
    lch = _lch(src.rgb)
    # Keep the source outline's hue but pull its darkness toward the reference outline.
    L = min(float(lch[0]), style.outline_L + 4.0, 18.0)
    C = min(float(lch[1]), style.outline_chroma) if lch[1] > 3 else style.outline_chroma * 0.4
    h = float(lch[2]) if lch[1] > 3 else COOL_HUE
    out = _rgb(np.array([L, C, h]))
    if config and config.palette_mode == "reference-guided" and style.adopt_colors and style.outline_colors:
        ref = _lch(style.outline_colors[0])
        if ref[0] <= 25:  # only adopt a reference outline that is itself a true dark outline
            out = _rgb(_blend_lch(_lch(out), ref, config.reference_weight))
    return out


def snap_to_palette(rgb: RGB, palette: list[RGB]) -> RGB:
    if not palette:
        return rgb
    lab = rgb_to_lab(np.array(rgb, dtype=np.uint8))
    pal = rgb_to_lab(np.array(palette, dtype=np.uint8))
    return palette[int(np.argmin(np.linalg.norm(pal - lab, axis=-1)))]


def build_palette(infos: list[ColorInfo], families: list[Family], ref: ReferenceStyle, config: RevampConfig) -> tuple[RGB, list[Ramp]]:
    style = _blend_style(ref, config)
    ramps: list[Ramp] = []
    # Trainers are frequently redesigned between generations; never re-hue their colours.
    dominant_id = max(families, key=lambda f: f.pixel_count).id if families and config.kind != "trainer" else None
    for f in families:
        ramp = build_ramp(f.base.rgb, f, style, config, dominant=(f.id == dominant_id))
        if config.palette_mode == "reference-palette" and style.palette:
            ramp = Ramp(
                source=ramp.source,
                deep=snap_to_palette(ramp.deep, style.palette),
                shadow=snap_to_palette(ramp.shadow, style.palette),
                base=snap_to_palette(ramp.base, style.palette),
                light=snap_to_palette(ramp.light, style.palette),
                highlight=snap_to_palette(ramp.highlight, style.palette),
                line=snap_to_palette(ramp.line, style.palette),
            )
        f.ramp = ramp
        ramps.append(ramp)
    outline = outline_color(infos, style, config)
    if config.palette_mode == "reference-palette" and style.palette:
        outline = snap_to_palette(outline, style.palette)
    return outline, ramps


def enforce_color_limit(rgba: np.ndarray, max_opaque: int, protected: list[RGB]) -> tuple[np.ndarray, list[str]]:
    """Merge the perceptually closest non-protected colour pairs until the limit is met."""
    warnings: list[str] = []
    out = rgba.copy()
    opaque = out[..., 3] > 0
    while True:
        colors = np.unique(out[opaque][:, :3], axis=0)
        if len(colors) <= max_opaque:
            break
        lab = rgb_to_lab(colors)
        counts = np.array([int(np.all(out[..., :3] == c, axis=-1)[opaque].sum()) for c in colors])
        prot = np.array([tuple(int(v) for v in c) in protected for c in colors])
        best: tuple[float, int, int] | None = None
        for i in range(len(colors)):
            for j in range(len(colors)):
                if i == j or prot[i]:
                    continue
                d = float(np.linalg.norm(lab[i] - lab[j]))
                # Prefer removing rare colours: weight distance by the removed colour's share.
                score = d * (0.5 + counts[i] / max(1, counts.sum()))
                if best is None or score < best[0]:
                    best = (score, i, j)
        if best is None:
            warnings.append("could not reduce palette without touching protected colours")
            break
        _, i, j = best
        mask = np.all(out[..., :3] == colors[i], axis=-1) & opaque
        out[mask, :3] = colors[j]
        warnings.append(f"merged colour {tuple(int(v) for v in colors[i])} into {tuple(int(v) for v in colors[j])}")
    return out, warnings


def palette_preview(colors: list[RGB], cell: int = 12) -> np.ndarray:
    n = max(1, len(colors))
    img = np.zeros((cell, cell * n, 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        img[:, i * cell : (i + 1) * cell, :3] = c
        img[:, i * cell : (i + 1) * cell, 3] = 255
    return img
