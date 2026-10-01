"""Color-space helpers: sRGB <-> OKLab, hex parsing, CIEDE2000."""

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def parse_hex(value: str) -> np.ndarray:
    """'#FF8800' or 'ff8800' -> float RGB in [0, 1]."""
    h = value.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"not a hex color: {value!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def to_hex(rgb) -> str:
    r, g, b = (np.clip(np.asarray(rgb, dtype=np.float64), 0, 1) * 255).round().astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """float sRGB [0,1] (..., 3) -> OKLab (..., 3)."""
    lms = _srgb_to_linear(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> float sRGB, clipped to [0, 1]."""
    lms_ = lab @ np.linalg.inv(_M2).T
    rgb_lin = (lms_ ** 3) @ np.linalg.inv(_M1).T
    return np.clip(_linear_to_srgb(rgb_lin), 0, 1)


def chroma_hue(lab: np.ndarray):
    """OKLab -> (chroma, hue radians)."""
    return np.hypot(lab[..., 1], lab[..., 2]), np.arctan2(lab[..., 2], lab[..., 1])


def delta_e_2000(rgb_a, rgb_b) -> np.ndarray:
    """CIEDE2000 between float sRGB colors (broadcastable)."""
    a = rgb2lab(np.atleast_2d(np.asarray(rgb_a, dtype=np.float64))[None])
    b = rgb2lab(np.atleast_2d(np.asarray(rgb_b, dtype=np.float64))[None])
    return deltaE_ciede2000(a, b)[0]
