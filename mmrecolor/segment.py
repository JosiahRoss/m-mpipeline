"""Candy masks.

P0 ships a classical, CPU-only segmenter: classify pixels by OKLab hue/chroma
against known candy colors, clean up with morphology, filter blobs by size.
It will also catch same-colored non-candy things (a pumpkin, a mascot), so use
--exclude boxes or supply masks from SAM / a paint tool via --mask. The P1+
plan replaces this with SAM 2/3 behind the same `segment()` signature.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .color import chroma_hue, parse_hex, srgb_to_oklab

# Approximate sRGB of each M&M shell color under neutral light.
CANDY_COLORS = {
    "red": "#B5221E",
    "orange": "#F26A1B",
    "yellow": "#FFD21F",
    "green": "#2FA84F",
    "blue": "#1E8FD6",
    "brown": "#5A3420",
    "black": "#16181A",
}


@dataclass
class ClassRule:
    hue: float            # OKLab hue center, radians
    hue_tol: float        # +/- radians
    min_chroma: float
    l_range: tuple        # (lo, hi) OKLab L
    achromatic: bool = False

    def loose(self) -> "ClassRule":
        """Wider window used to grow seeds over a candy's shaded/blurred edge."""
        if self.achromatic:
            return ClassRule(0.0, np.pi, 0.0, (0.0, self.l_range[1] + 0.06), achromatic=True)
        return ClassRule(self.hue, self.hue_tol * 1.4, self.min_chroma * 0.5,
                         (max(0.0, self.l_range[0] - 0.12), min(1.0, self.l_range[1] + 0.06)))


def default_rule(name: str) -> ClassRule:
    lab = srgb_to_oklab(parse_hex(CANDY_COLORS[name]))
    c, h = chroma_hue(lab)
    if name == "black":
        return ClassRule(0.0, np.pi, 0.0, (0.0, 0.30), achromatic=True)
    if name == "brown":
        return ClassRule(float(h), 0.45, 0.04, (0.20, 0.50))
    # Red, orange and skin/wood/dough tones crowd 20-60 deg, so keep these tight.
    tol = {"red": 0.20, "orange": 0.16}.get(name, 0.45)
    return ClassRule(float(h), tol, max(0.10, float(c) * 0.70), (0.30, 0.90))


def rule_from_pick(rgb: np.ndarray, x: int, y: int, radius: int = 3) -> ClassRule:
    """Build a rule from one clicked candy: median color of a small patch.

    This is the classical stand-in for a SAM point prompt, and adapts to the
    shot's grade instead of nominal shell colors.
    """
    patch = rgb[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    lab = np.median(srgb_to_oklab(patch.reshape(-1, 3)), axis=0)
    c, h = chroma_hue(lab)
    L = float(lab[0])
    if c < 0.05:
        return ClassRule(0.0, np.pi, 0.0, (0.0, L + 0.12), achromatic=True)
    return ClassRule(float(h), 0.17, float(c) * 0.55, (max(0.0, L - 0.22), min(1.0, L + 0.30)))


def _hue_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def candy_radius(shape) -> float:
    """Smallest plausible candy radius in px, scaled to the frame."""
    return max(2.0, 0.0014 * float(np.hypot(*shape[:2])))


def thick_seeds(seeds: np.ndarray, radius: float) -> np.ndarray:
    """Keep only seed pixels inside a disk of `radius`: drops thin same-hue
    structures (a cookie's baked rim, a sleeve edge) that aren't candy-shaped."""
    r = max(1, int(round(radius)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)


def grow(seeds: np.ndarray, loose: np.ndarray) -> np.ndarray:
    """Hysteresis: keep loose-mask components that touch a strict seed."""
    n, lab, _, _ = cv2.connectedComponentsWithStats((loose | seeds).astype(np.uint8), connectivity=8)
    hit = np.zeros(n, dtype=bool)
    hit[np.unique(lab[seeds])] = True
    hit[0] = False
    return hit[lab]


def classify(lab: np.ndarray, rule: ClassRule) -> np.ndarray:
    L = lab[..., 0]
    c, h = chroma_hue(lab)
    m = (L >= rule.l_range[0]) & (L <= rule.l_range[1])
    if rule.achromatic:
        return m & (c < 0.05)
    return m & (c >= rule.min_chroma) & (_hue_dist(h, rule.hue) <= rule.hue_tol)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (the printed 'm', specular dots) inside blobs."""
    inv = (~binary).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = binary.shape
    out = binary.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and area < 0.002 * h * w:
            out[lab == i] = True
    return out


def segment_many(rgb: np.ndarray, rules: dict, **kw) -> dict:
    """Masks for several colors; a pixel matching two hue windows goes to the
    closer hue, so red and orange masks never overlap."""
    lab = srgb_to_oklab(rgb)
    _, hue = chroma_hue(lab)
    chromatic = [n for n, r in rules.items() if not r.achromatic]
    nearest = None
    if len(chromatic) > 1:
        nearest = np.argmin(np.stack([_hue_dist(hue, rules[n].hue) for n in chromatic]), axis=0)
    raw = {}
    for n, r in rules.items():
        seeds, loose = classify(lab, r), classify(lab, r.loose())
        seeds = thick_seeds(seeds, candy_radius(rgb.shape))
        if nearest is not None and n in chromatic:
            own = nearest == chromatic.index(n)
            seeds, loose = seeds & own, loose & own
        raw[n] = grow(seeds, loose)
    return {n: segment(rgb, n, raw=raw[n], **kw) for n in rules}


def segment(rgb: np.ndarray, name: str, *, exclude=(), min_area_frac=2e-5,
            max_area_frac=0.02, max_elongation=8.0, feather=1.5,
            rule: ClassRule | None = None, raw=None):
    """Soft mask in [0, 1] for candies of color `name`.

    exclude: iterable of (x, y, w, h) boxes to zero out.
    max_area_frac: drop single blobs bigger than this share of the frame
    (pumpkins, bags). Piles of touching candies can exceed it; raise it then.
    raw: precomputed boolean pixel classification (from segment_many).
    """
    if raw is None:
        rule = rule or default_rule(name)
        lab = srgb_to_oklab(rgb)
        raw = grow(thick_seeds(classify(lab, rule), candy_radius(rgb.shape)),
                   classify(lab, rule.loose()))
    m = raw.copy()
    for x, y, w, h in exclude:
        m[y:y + h, x:x + w] = False

    u8 = m.astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k3)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    total = m.shape[0] * m.shape[1]
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep[1:] = (areas[1:] >= min_area_frac * total) & (areas[1:] <= max_area_frac * total)
    # Elongation: area vs. the blob's largest inscribed disk. A candy is ~1-2,
    # a pile of candies stays well under the limit, a thin arc (baked cookie
    # rim, finger edge) blows past it.
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 5)
    max_r = np.zeros(n)
    np.maximum.at(max_r, lab_cc.ravel(), dist.ravel())
    with np.errstate(divide="ignore"):
        elong = areas / (np.pi * np.maximum(max_r, 0.5) ** 2)
    keep &= elong <= max_elongation
    binary = _fill_holes(keep[lab_cc])

    alpha = binary.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        # Blur must not leak outside a 2px dilation of the mask.
        grown = cv2.dilate(binary.astype(np.uint8), k3, iterations=2).astype(bool)
        alpha[~grown] = 0
    return np.clip(alpha, 0, 1)


def load_mask(path: str, shape) -> np.ndarray:
    """Load an external mask (white = candy) e.g. exported from SAM."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m.astype(np.float32) / 255.0
