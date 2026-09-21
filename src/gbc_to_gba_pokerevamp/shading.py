"""Form-following shade synthesis: a smooth per-region illumination field is quantised into
shade levels, then thresholded so the amount of each level matches what Gen III references do.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.geometry import dilate, erode, illumination, small_components
from gbc_to_gba_pokerevamp.models import ReferenceStyle
from gbc_to_gba_pokerevamp.palette import Family


def _quantile_threshold(values: np.ndarray, fraction: float, upper: bool) -> float:
    if len(values) == 0:
        return np.inf if upper else -np.inf
    q = 1.0 - fraction if upper else fraction
    return float(np.quantile(values, np.clip(q, 0.0, 1.0)))


def _clean(mask: np.ndarray, allowed: np.ndarray, min_area: int) -> np.ndarray:
    """Turn a thresholded field into deliberate pixel clusters: no 1px strips, no specks."""
    out = mask & allowed
    # Opening with a 2x2 kernel removes single-pixel strips while keeping 2px-wide bands.
    opened = ndimage.binary_opening(out, structure=np.ones((2, 2), bool))
    # Keep 1px runs only where they hug a wider part of the same region (inner anti-aliasing).
    hugging = out & ndimage.binary_dilation(opened, structure=np.ones((3, 3), bool))
    out = (opened | hugging) & allowed
    out = ndimage.binary_closing(out, structure=np.ones((2, 2), bool)) & allowed
    return out & ~small_components(out, min_area - 1)


def synthesize_shading(
    mask: np.ndarray,
    family_map: np.ndarray,
    level: np.ndarray,
    protected: np.ndarray,
    families: list[Family],
    config: RevampConfig,
    style: ReferenceStyle,
) -> np.ndarray:
    """Return a new level map with shadow/deep/light/highlight regions per family.

    Source shadows are kept as evidence: pixels the artist already darkened stay at least at
    shadow level. New levels come from thresholding the illumination field so that the share
    of shadow/light pixels matches the reference style (scaled by shading_strength).
    """
    out = level.copy()
    strength = config.shading_strength
    if strength <= 0:
        return out
    light = (config.light_x, config.light_y)
    s = 0.4 + 0.6 * strength

    for f in families:
        region = (family_map == f.id) & ~protected
        area = int(region.sum())
        if area < 24:
            continue
        lit = illumination(region, mask, light)

        # Rule 1 of revamping: keep the source shading only where it matches the target style.
        # Gen I/II artists often flooded a sprite with its one dark colour; if the source has
        # far more shadow than the reference would, the most-lit shadow pixels revert to base.
        want_shadow = style.shadow_fraction * s
        src_shadow = region & (out < 0)
        n_shadow = int(src_shadow.sum())
        if n_shadow > want_shadow * area * 1.3 and n_shadow > 8:
            keep = int(round(want_shadow * area * 1.15))
            t_keep = float(np.sort(lit[src_shadow])[min(keep, n_shadow - 1)])
            demote = src_shadow & (lit > t_keep)
            demote = _clean(demote, src_shadow, 4)
            out[demote] = 0

        editable = region & (out == 0)  # only base pixels are re-shaded; source shades stay
        vals = lit[editable]
        n_edit = len(vals)
        if n_edit < 8:
            continue

        # Fractions are of the whole region; convert to fractions of the editable base pixels
        # after crediting what the source already provides.
        have_shadow = int((region & (out < 0)).sum())
        have_light = int((region & (out > 0)).sum())
        frac_shadow = max(0.0, want_shadow * area - have_shadow) / n_edit
        want_deep = min(style.deep_fraction * s, want_shadow * 0.45)
        want_light = style.light_fraction * s
        frac_light = max(0.0, want_light * area - have_light) / n_edit
        want_high = min(style.highlight_fraction * config.highlight_strength * 2.0, want_light * 0.5)
        frac_high = want_high * area / n_edit
        if f.achromatic:
            frac_light *= 0.4
            want_high = frac_high = 0.0

        t_shadow = _quantile_threshold(vals, frac_shadow, upper=False)
        t_light = _quantile_threshold(vals, frac_light, upper=True)
        t_high = _quantile_threshold(vals, frac_high, upper=True)

        min_area = 4 if area < 200 else 6
        shadow = _clean(editable & (lit <= t_shadow), editable, min_area)
        out[shadow] = -1
        if strength > 0.3 and want_deep > 0:
            shadow_all = region & (out < 0)
            t_deep = _quantile_threshold(lit[shadow_all], min(1.0, want_deep / max(want_shadow, 1e-6)), upper=False)
            deep = shadow_all & (lit <= t_deep) & (out == -1)
            # Deep shadow only where the shadow band is thick enough to hold two tones.
            thick = dilate(erode(shadow_all, 1, connectivity=8), 1, connectivity=8)
            deep = _clean(deep & thick, shadow_all & (out == -1), min_area)
            out[deep] = -2
        light_px = _clean(editable & (out == 0) & (lit >= t_light), editable & (out == 0), min_area)
        out[light_px] = 1
        if want_high > 0 and area >= 150:
            hi = _clean(light_px & (lit >= t_high), light_px, 3)
            out[hi] = 2
    return out
