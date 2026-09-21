"""Vectorised sRGB <-> CIELAB / LCh conversions (D65) implemented with NumPy only."""

from __future__ import annotations

import numpy as np

_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) uint8 -> (..., 3) float Lab."""
    lin = srgb_to_linear(np.asarray(rgb))
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f**3 > eps, f**3, (116 * f - 16) / kappa) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    return linear_to_srgb(lin)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], C, h], axis=-1)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    hr = np.radians(lch[..., 2])
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(hr), lch[..., 1] * np.sin(hr)], axis=-1)


def rgb_to_lch(rgb: np.ndarray) -> np.ndarray:
    return lab_to_lch(rgb_to_lab(rgb))


def lch_to_rgb(lch: np.ndarray) -> np.ndarray:
    return lab_to_rgb(lch_to_lab(lch))


def hue_delta(h_from: float, h_to: float) -> float:
    """Signed shortest arc from h_from to h_to, in (-180, 180]."""
    d = (h_to - h_from + 180.0) % 360.0 - 180.0
    return d


def shift_hue_toward(h: float, target: float, degrees: float) -> float:
    """Move hue `h` toward `target` by at most `degrees` along the shortest arc."""
    d = hue_delta(h, target)
    step = float(np.sign(d)) * min(abs(d), degrees)
    return (h + step) % 360.0


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(lab_a) - np.asarray(lab_b), axis=-1)
