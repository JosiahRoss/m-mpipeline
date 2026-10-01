"""Extract dominant colors from a dropped color reference (swatch, chip, photo)."""

import cv2
import numpy as np

from .color import chroma_hue, oklab_to_srgb, srgb_to_oklab, to_hex


def extract_palette(rgb: np.ndarray, k: int = 5, min_chroma: float = 0.0,
                    max_side: int = 256, seed: int = 0):
    """k-means in OKLab. Returns [{'hex', 'share'}] sorted by pixel share.

    min_chroma drops near-neutral clusters (paper white, table grey) so a photo
    of a swatch on a desk still yields the swatch color first.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lab = srgb_to_oklab(rgb.reshape(-1, 3)).astype(np.float32)
    k = max(1, min(k, len(lab)))
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    out = []
    for i in np.argsort(-shares):
        c, _ = chroma_hue(centers[i].astype(np.float64))
        if c < min_chroma:
            continue
        out.append({"hex": to_hex(oklab_to_srgb(centers[i].astype(np.float64))),
                    "share": round(float(shares[i]), 4)})
    return out
