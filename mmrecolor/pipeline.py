"""End-to-end still-image job: segment -> recolor -> measure -> report."""

import cv2
import numpy as np
from PIL import Image

from .recolor import measure, recolor
from .segment import default_rule, load_mask, rule_from_pick, segment_many


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def save_rgb(path, rgb):
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(path)


def run(image, mapping, *, masks=None, picks=None, exclude=(), max_area_frac=0.02,
        min_area_frac=2e-5, tolerance=3.0):
    """mapping: {source_color_name: target_hex}.
    masks: {name: path} external masks (SAM, paint tool) that skip segmentation.
    picks: {name: (x, y)} a clicked candy whose color defines that class.

    Returns (out_rgb, {name: alpha}, report).
    """
    masks, picks = masks or {}, picks or {}
    rules = {n: rule_from_pick(image, *picks[n]) if n in picks else default_rule(n)
             for n in mapping if n not in masks}
    alphas = segment_many(image, rules, exclude=exclude, max_area_frac=max_area_frac,
                          min_area_frac=min_area_frac) if rules else {}
    alphas.update({n: load_mask(p, image.shape) for n, p in masks.items() if n in mapping})
    out = image.copy()
    rows = []
    for name, target in mapping.items():
        alpha = alphas[name]
        # Masks come from the original frame; recolor sequentially on `out`.
        out, info = recolor(out, alpha, target)
        m = measure(out, alpha, target) or {}
        n_blobs = cv2.connectedComponents((alpha >= 0.5).astype(np.uint8))[0] - 1
        de = m.get("delta_e_2000")
        rows.append({"source": name, "target_hex": target.upper() if target.startswith("#") else "#" + target.upper(),
                     "mask_source": "file" if name in masks else ("pick" if name in picks else "classical"), "blobs": int(n_blobs),
                     **info, **m, "pass": de is not None and de <= tolerance})
    return out, alphas, {"tolerance_delta_e": tolerance, "mappings": rows,
                         "pass": all(r["pass"] for r in rows)}


def overlay(image, alphas):
    """Debug view: each mask tinted a distinct color over a darkened frame."""
    tints = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 1, 0), (0, 0.5, 1)]
    out = image * 0.45
    for i, a in enumerate(alphas.values()):
        t = np.array(tints[i % len(tints)])
        out = out * (1 - a[..., None]) + (0.35 * image + 0.65 * t) * a[..., None]
    return out


def side_by_side(before, after):
    gap = np.ones((before.shape[0], 8, 3))
    return np.concatenate([before, gap, after], axis=1)
