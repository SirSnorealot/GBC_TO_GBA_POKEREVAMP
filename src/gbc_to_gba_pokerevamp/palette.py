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
REFERENCE_MATCH_HUE = 35.0  # max hue distance for adopting a reference family's colors


@dataclass
class Family:
    id: int
    colors: list[ColorInfo]
    base: ColorInfo
    levels: dict[RGB, int] = field(default_factory=dict)  # source color -> ramp level
    achromatic: bool = False
    ramp: Ramp | None = None
    level_offset: int = 0  # reference level this family's base maps to (set by build_palette)
    shade_of: int | None = None  # dominant family id when this family is its GBC shading color
    forced_ref: dict | None = None  # reference family chosen by position (same-species only)

    @property
    def pixel_count(self) -> int:
        return sum(c.count for c in self.colors)


def group_families(infos: list[ColorInfo], kind: str = "unknown", step_levels: bool = False) -> list[Family]:
    """Cluster non-line source colors into hue families; each family becomes one shade ramp.

    `step_levels=True` (used for 16-color references) assigns levels by lightness distance
    from the base so two near-identical shades share a level instead of eating the light and
    highlight slots; GBC sources with 1-2 colors per family use plain rank order.
    """
    tolerance = FAMILY_HUE_TOLERANCE if kind != "trainer" else 28.0
    if step_levels:
        # Gen III ramps: lit colors stay close in hue, shadows swing hard toward blue/red.
        tolerance = 24.0
    locals_ = [c for c in infos if c.role not in (ColorRole.OUTLINE, ColorRole.INTERNAL_LINE)]
    if step_levels:
        # Seed families with the lit colors so hue-shifted shadows attach to them, not vice versa.
        locals_.sort(key=lambda c: (-c.L, -c.count))
    else:
        locals_.sort(key=lambda c: -c.count)
    families: list[Family] = []
    for c in locals_:
        achro = c.chroma < ACHROMATIC_CHROMA
        best_f: Family | None = None
        best_d = np.inf
        for f in families:
            if achro != f.achromatic:
                continue
            if achro:
                # Whites/grays join a single gray family only when both are light; dark grays stay separate.
                if (c.L > 60) == (f.base.L > 60):
                    best_f, best_d = f, 0.0
                    break
                continue
            d = abs((c.hue - f.base.hue + 180.0) % 360.0 - 180.0)
            if step_levels:
                darker_by = f.base.L - c.L
                slack = tolerance + (24.0 if darker_by > 12.0 else 0.0)
                if c.chroma < 20.0 and f.base.chroma < 20.0:
                    slack += 20.0
            else:
                # GBC shadow colors often sit far around the hue wheel from the body color
                # (Pikachu's red-orange under yellow), so clearly darker colors get more slack.
                slack = tolerance + 30.0 if c.L < f.base.L - 20.0 else tolerance
                # Hue is unreliable for low-chroma colors.
                slack += 1.5 * max(0.0, 25.0 - min(c.chroma, f.base.chroma))
            # Join the *closest* eligible family, not the first: a teal must not be captured by
            # a yellow-green just because that family was seeded earlier.
            if d <= slack and d < best_d:
                best_f, best_d = f, d
        if best_f is not None:
            best_f.colors.append(c)
        else:
            families.append(Family(id=len(families), colors=[c], base=c, achromatic=achro))
    for f in families:
        if step_levels:
            # Gen III artists often paint more shadow than base; the base is the lightest color
            # that still covers a major share of the family, not simply the most frequent one.
            top = max(c.count for c in f.colors)
            major = [c for c in f.colors if c.count >= 0.5 * top]
            f.base = max(major, key=lambda c: c.L)
        else:
            f.base = max(f.colors, key=lambda c: c.count)
        f.levels[f.base.rgb] = 0
        if step_levels and len(f.colors) > 2:
            # Cluster shades within 5 L of each other, then rank clusters away from the base,
            # so near-identical shades share a level and real tones fill -2..2 in order.
            ordered = sorted(f.colors, key=lambda c: c.L)
            clusters: list[list[ColorInfo]] = [[ordered[0]]]
            for c in ordered[1:]:
                if c.L - clusters[-1][-1].L < 5.0:
                    clusters[-1].append(c)
                else:
                    clusters.append([c])
            base_idx = next(i for i, cl in enumerate(clusters) if f.base in cl)
            for i, cl in enumerate(clusters):
                for c in cl:
                    f.levels[c.rgb] = int(np.clip(i - base_idx, -2, 2))
            continue
        lighter = sorted([c for c in f.colors if c.L > f.base.L], key=lambda c: c.L)
        darker = sorted([c for c in f.colors if c.L < f.base.L], key=lambda c: -c.L)
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
    """Interpolate two LCh colors, taking the shortest hue arc."""
    h = (float(a[2]) + w * hue_delta(float(a[2]), float(b[2]))) % 360.0
    return np.array([(1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1], h])


def match_reference_family(
    family: Family, style: ReferenceStyle, dominant: bool = False, centroid: tuple[float, float] | None = None
) -> dict | None:
    """Find the reference color family with the same role as this source family.

    Same-hue matches come first (yellow->yellow, white->white). For a same-species reference a
    secondary color that has no hue match (Totodile's red stripe vs FRLG's yellow one) falls
    back to the reference region sitting in the same *place* on the body.
    """
    best, best_score = None, 0.0
    top_weight = max((rf["weight"] for rf in style.families), default=0.0)
    if style.same_subject and dominant:
        # Same species: the main body color maps to the official main body color whatever
        # its hue (Yellow's near-white Pikachu -> FRLG yellow, green Bulbasaur -> teal).
        cands = [rf for rf in style.families if rf["weight"] >= top_weight - 1e-9]
        return max(cands, key=lambda rf: rf["pixels"]) if cands else None
    for rf in style.families:
        # The best-ranked reference always wins; pixel share only breaks ties within one reference.
        rank_score = rf["weight"] * 1000.0
        share = rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"]))
        ref_L, ref_C, ref_h = rf["base_lch"]
        if family.achromatic:
            if family.base.L > 60:
                # Whites map to the reference's white if it has one, else to a warm cream/pale
                # yellow region (Gen III bellies); never to a pale body color such as Squirtle's blue.
                if rf["achromatic"] and ref_L >= 75 and share >= 0.008:
                    score = rank_score + 10.0 + share
                elif ref_L >= 75 and ref_C <= 65 and 35.0 <= ref_h <= 110.0 and share >= 0.03:
                    score = rank_score + share
                else:
                    continue
            elif style.same_subject and ref_L <= 45 and share >= 0.03:
                # Thick GBC black (Cyndaquil's back, Snorlax's body) takes the official dark
                # body color of the same species, whatever its hue.
                score = rank_score + share + (5.0 if not rf["achromatic"] else 0.0)
            elif not (rf["achromatic"] and ref_L <= 60):
                continue
            else:
                score = rank_score + share
        else:
            if rf["achromatic"]:
                continue
            d = abs(hue_delta(family.base.hue, ref_h))
            if d > REFERENCE_MATCH_HUE:
                continue
            score = rank_score + share * (1.0 - d / (2 * REFERENCE_MATCH_HUE))
        if score > best_score:
            best, best_score = rf, score
    if best is None and style.same_subject and centroid is not None and not family.achromatic:
        dom = max((rf for rf in style.families if rf["weight"] >= top_weight - 1e-9), key=lambda rf: rf["pixels"], default=None)
        best_d = 0.22
        for rf in style.families:
            if rf is dom or rf["achromatic"] or rf.get("centroid") is None or rf["weight"] < top_weight - 1e-9:
                continue
            if rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"])) < 0.015:
                continue
            d = float(np.hypot(rf["centroid"][0] - centroid[0], rf["centroid"][1] - centroid[1]))
            if d < best_d:
                best, best_d = rf, d
    return best


def build_ramp(
    base_rgb: RGB,
    family: Family,
    style: ReferenceStyle,
    config: RevampConfig,
    dominant: bool = False,
    ref_family: dict | None = None,
    level_offset: int = 0,
) -> Ramp:
    """Derive deep/shadow/base/light/highlight/line from a source color in LCh space.

    Existing source shades are kept where the family already has them; missing levels are
    synthesized from the (blended) reference deltas. In reference-guided mode every level is
    then pulled toward the matched reference family by `reference_weight`. `level_offset`
    shifts which reference level a source level maps to (a GBC "shading color" family maps
    onto the reference's shadow tones, not its base).
    """
    base = _lch(base_rgb)
    L, C, h = float(base[0]), float(base[1]), float(base[2])
    s = config.shading_strength * 0.5 + 0.5  # keep some separation even at low strength
    hs = config.hue_shift_strength if not family.achromatic else 0.0
    if config.kind == "trainer":
        hs *= 0.5
    if family.achromatic:
        C = 0.0

    if ref_family is None and config.palette_mode == "reference-guided" and style.adopt_colors:
        ref_family = match_reference_family(family, style, dominant)
    # A same-species reference is the authoritative palette ("switch out the colors"); a
    # user-chosen reference pulls by reference_weight; auto-picked shape-alikes never recolor.
    w = (1.0 if style.same_subject else config.reference_weight) if ref_family else 0.0
    if config.kind == "trainer" and not family.achromatic:
        w *= 0.5  # skin/hair/clothing hues must stay closer to the source

    def ref_at(level: int) -> RGB | None:
        """Reference color for a level, falling back to the nearest level that exists."""
        if not ref_family:
            return None
        levels = ref_family["levels"]
        target = level + level_offset
        for cand in sorted(range(-2, 3), key=lambda v: (abs(v - target), v)):
            if str(cand) in levels and abs(cand - target) <= 1:
                return tuple(levels[str(cand)])
        return None

    def adapt(rgb: RGB, level: int) -> RGB:
        ref_rgb = ref_at(level)
        if ref_rgb is None or w <= 0:
            return rgb
        return _rgb(_blend_lch(_lch(rgb), _lch(ref_rgb), w))

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
    # Outline base: a dark shade of the local color that still reads as a line against every
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


def build_palette(
    infos: list[ColorInfo],
    families: list[Family],
    ref: ReferenceStyle,
    config: RevampConfig,
    centroids: dict[int, tuple[float, float]] | None = None,
) -> tuple[RGB, list[Ramp]]:
    style = _blend_style(ref, config)
    ramps: list[Ramp] = []
    centroids = centroids or {}
    # Trainers are frequently redesigned between generations; never re-hue their colors.
    dominant = max(families, key=lambda f: f.pixel_count) if families and config.kind != "trainer" else None
    guided = config.palette_mode == "reference-guided" and style.adopt_colors
    dom_ref = match_reference_family(dominant, style, True) if (dominant and guided) else None
    for f in families:
        ref_family = None
        offset = 0
        if guided:
            is_dom = dominant is not None and f.id == dominant.id
            if f.forced_ref is not None:
                ref_family = f.forced_ref
            else:
                ref_family = dom_ref if is_dom else match_reference_family(f, style, False, centroids.get(f.id))
            if (
                style.same_subject and not is_dom and dominant is not None and dom_ref is not None
                and not f.achromatic and ref_family is dom_ref
            ):
                # A second family in the body's hue is the GBC shading color (Yellow Pikachu's
                # gold under near-white): map it onto the reference's darker/lighter levels.
                if f.base.L < dominant.base.L - 8.0:
                    offset = -1
                    f.shade_of = dominant.id
                elif f.base.L > dominant.base.L + 8.0:
                    offset = 1
        f.level_offset = offset
        ramp = build_ramp(f.base.rgb, f, style, config, dominant=(dominant is not None and f.id == dominant.id), ref_family=ref_family, level_offset=offset)
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
    """Merge the perceptually closest non-protected color pairs until the limit is met."""
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
                # Prefer removing rare colors: weight distance by the removed color's share.
                score = d * (0.5 + counts[i] / max(1, counts.sum()))
                if best is None or score < best[0]:
                    best = (score, i, j)
        if best is None:
            warnings.append("could not reduce palette without touching protected colors")
            break
        _, i, j = best
        mask = np.all(out[..., :3] == colors[i], axis=-1) & opaque
        out[mask, :3] = colors[j]
        warnings.append(f"merged color {tuple(int(v) for v in colors[i])} into {tuple(int(v) for v in colors[j])}")
    return out, warnings


def palette_preview(colors: list[RGB], cell: int = 12) -> np.ndarray:
    n = max(1, len(colors))
    img = np.zeros((cell, cell * n, 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        img[:, i * cell : (i + 1) * cell, :3] = c
        img[:, i * cell : (i + 1) * cell, 3] = 255
    return img
