"""End-to-end still-image job: segment -> recolor -> measure -> report."""

import cv2
import numpy as np
from PIL import Image

from .recolor import measure, recolor
from .color import srgb_to_oklab
from .segment import (candy_radius, classify, default_rule, load_mask, rule_from_pick,
                      segment_many, thick_seeds)


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def save_rgb(path, rgb):
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(path)


def find_missed(out, rule, exclude=(), covered=None, limit=20):
    """Candy-shaped blobs still in the source color after recoloring."""
    seeds = thick_seeds(classify(srgb_to_oklab(out), rule), candy_radius(out.shape))
    for x, y, w, h in exclude:
        seeds[y:y + h, x:x + w] = False
    if covered is not None:
        seeds &= ~covered
    n, _, stats, cents = cv2.connectedComponentsWithStats(seeds.astype(np.uint8))
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:limit] + 1
    return [{"x": int(cents[i][0]), "y": int(cents[i][1]), "area": int(stats[i, cv2.CC_STAT_AREA])}
            for i in order]


def run(image, mapping, *, masks=None, picks=None, exclude=(), protect=(), max_area_frac=0.02,
        min_area_frac=2e-5, tolerance=3.0, segmenter="classical", predictor=None, rounds=3):
    """mapping: {source_color_name: target_hex}.
    masks: {name: path} external masks (SAM, paint tool) that skip segmentation.
    picks: {name: (x, y)} a clicked candy whose color defines that class.
    segmenter: "classical" or "sam" (predictor: a Sam2Predictor or test double).
    protect: [(x, y, w, h)] boxes around objects to leave untouched, e.g. the bag (SAM only).

    Returns (out_rgb, {name: alpha}, report).
    """
    masks, picks = masks or {}, picks or {}
    rules = {n: rule_from_pick(image, *picks[n]) if n in picks else default_rule(n)
             for n in mapping if n not in masks}
    sam_info, blocked = {}, None
    if protect and segmenter != "sam":
        raise ValueError("protect points need segmenter='sam'; use exclude boxes instead")
    if segmenter == "sam" and rules:
        from .sam import Sam2Predictor, nearest_hue_masks, protect_mask, sam_segment

        predictor = predictor or Sam2Predictor()
        predictor.set_image(image)
        blocked = protect_mask(predictor, protect) if protect else None
        owns = nearest_hue_masks(image, rules)
        alphas = {}
        for n, rule in rules.items():
            r = sam_segment(image, rule, predictor, own=owns.get(n), exclude=exclude, blocked=blocked,
                            pick=picks.get(n), rounds=rounds)
            alphas[n] = r.alpha
            sam_info[n] = {"ref_area": int(r.ref_area), "accepted": len(r.accepted),
                           "rejected": r.rejected[:30]}
    elif rules:
        alphas = segment_many(image, rules, exclude=exclude, max_area_frac=max_area_frac,
                              min_area_frac=min_area_frac)
    else:
        alphas = {}
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
        source = "file" if name in masks else (segmenter + ("+pick" if name in picks else ""))
        row = {"source": name, "target_hex": target.upper() if target.startswith("#") else "#" + target.upper(),
               "mask_source": source, "blobs": int(n_blobs), **info, **m,
               "pass": de is not None and de <= tolerance}
        if name in rules:
            row["missed"] = find_missed(out, rules[name], exclude, covered=blocked)
        if name in sam_info:
            row["sam"] = sam_info[name]
        rows.append(row)
    if blocked is not None:
        alphas["protected"] = blocked.astype(np.float32)
    return out, alphas, {"tolerance_delta_e": tolerance, "segmenter": segmenter, "mappings": rows,
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
