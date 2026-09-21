"""Form-following shade synthesis.

Each connected sub-form of a color region (head, arm, tail...) is shaded on its own like a
hand-drawn Gen III sprite: a smooth illumination field over the sub-form is thresholded so the
share of shadow / deep / light / highlight pixels matches what the reference does, then the
result is cleaned into deliberate 2px-wide-or-better clusters.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.analyze import EIGHT
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.geometry import dilate, erode, illumination, small_components
from gbc_to_gba_pokerevamp.models import ReferenceStyle
from gbc_to_gba_pokerevamp.palette import Family


def _quantile_threshold(values: np.ndarray, fraction: float, upper: bool) -> float:
    if len(values) == 0 or fraction <= 0:
        return np.inf if upper else -np.inf
    q = 1.0 - fraction if upper else fraction
    return float(np.quantile(values, np.clip(q, 0.0, 1.0)))


def _clean(mask: np.ndarray, allowed: np.ndarray, min_area: int) -> np.ndarray:
    """Turn a thresholded field into deliberate pixel clusters: no 1px strips, no specks."""
    out = mask & allowed
    opened = ndimage.binary_opening(out, structure=np.ones((2, 2), bool))
    # Keep 1px runs only where they hug a wider part of the same region (inner anti-aliasing).
    hugging = out & ndimage.binary_dilation(opened, structure=np.ones((3, 3), bool))
    out = (opened | hugging) & allowed
    out = ndimage.binary_closing(out, structure=np.ones((2, 2), bool)) & allowed
    return out & ~small_components(out, min_area - 1)


def _shade_subform(
    out: np.ndarray,
    sub: np.ndarray,
    lit: np.ndarray,
    achromatic: bool,
    config: RevampConfig,
    style: ReferenceStyle,
) -> None:
    strength = config.shading_strength
    s = 0.4 + 0.6 * strength
    area = int(sub.sum())
    if area < 12:
        return
    # Small sub-forms (fingers, toes, ear tips) get a single shadow tone, no highlights.
    small = area < 40
    min_area = 3 if small else (6 if area < 200 else 10)

    want_shadow = style.shadow_fraction * s
    # Rule 1 of revamping: keep the source shading only where it matches the target style.
    # Gen I/II artists often flooded a sprite with its one dark color; if the source has far
    # more shadow than the reference would, the most-lit shadow pixels revert to base. Deep
    # (-2) pixels are deliberate dark masses and are never demoted.
    src_shadow = sub & (out == -1)
    n_shadow = int((sub & (out < 0)).sum())
    if n_shadow > want_shadow * area * 1.3 and src_shadow.sum() > 8:
        keep = int(round(want_shadow * area * 1.15))
        vals_s = np.sort(lit[src_shadow])
        t_keep = float(vals_s[min(max(0, keep - int((sub & (out == -2)).sum())), len(vals_s) - 1)])
        demote = _clean(src_shadow & (lit > t_keep), src_shadow, 4)
        out[demote] = 0

    editable = sub & (out == 0)
    vals = lit[editable]
    n_edit = len(vals)
    if n_edit < 6:
        return
    have_shadow = int((sub & (out < 0)).sum())
    have_light = int((sub & (out > 0)).sum())
    frac_shadow = max(0.0, want_shadow * area - have_shadow) / n_edit
    want_deep = min(style.deep_fraction * s, want_shadow * 0.45)
    want_light = style.light_fraction * s
    frac_light = max(0.0, want_light * area - have_light) / n_edit
    want_high = min(style.highlight_fraction * config.highlight_strength * 2.0, want_light * 0.5)
    frac_high = want_high * area / n_edit
    if achromatic:
        frac_light *= 0.4
        want_high = frac_high = 0.0
    if small:
        frac_light = frac_high = 0.0
        want_deep = 0.0

    t_shadow = _quantile_threshold(vals, frac_shadow, upper=False)
    shadow = _clean(editable & (lit <= t_shadow), editable, min_area)
    out[shadow] = -1

    if strength > 0.3 and want_deep > 0:
        shadow_all = sub & (out < 0)
        if shadow_all.any():
            t_deep = _quantile_threshold(lit[shadow_all], min(1.0, want_deep / max(want_shadow, 1e-6)), upper=False)
            deep = shadow_all & (lit <= t_deep) & (out == -1)
            # Deep shadow only where the shadow band is thick enough to hold two tones.
            thick = dilate(erode(shadow_all, 1, connectivity=8), 1, connectivity=8)
            deep = _clean(deep & thick, shadow_all & (out == -1), min_area)
            out[deep] = -2

    base_now = editable & (out == 0)
    t_light = _quantile_threshold(lit[base_now] if base_now.any() else vals, frac_light, upper=True)
    light_px = _clean(base_now & (lit >= t_light), base_now, min_area)
    out[light_px] = 1
    if want_high > 0 and area >= 150 and light_px.any():
        t_high = _quantile_threshold(lit[light_px], min(1.0, frac_high / max(frac_light, 1e-6)), upper=True)
        hi = _clean(light_px & (lit >= t_high), light_px, 3)
        out[hi] = 2


def synthesize_shading(
    mask: np.ndarray,
    family_map: np.ndarray,
    level: np.ndarray,
    protected: np.ndarray,
    families: list[Family],
    config: RevampConfig,
    style: ReferenceStyle,
) -> np.ndarray:
    """Return a new level map with shadow/deep/light/highlight regions per family sub-form."""
    out = level.copy()
    if config.shading_strength <= 0:
        return out
    light = (config.light_x, config.light_y)

    for f in families:
        region = (family_map == f.id) & ~protected
        if region.sum() < 12:
            continue
        lit = illumination(region, mask, light)
        # Sub-forms are what remains when the linework is removed: each is shaded on its own.
        labels, n = ndimage.label(region, structure=EIGHT)
        for i in range(1, n + 1):
            _shade_subform(out, labels == i, lit, f.achromatic, config, style)
    return out
