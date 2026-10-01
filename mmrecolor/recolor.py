"""Shift masked candies to a target color, keeping shading, highlights and print."""

import numpy as np

from .color import chroma_hue, delta_e_2000, oklab_to_srgb, parse_hex, srgb_to_oklab, to_hex

ACHROMATIC_C = 0.04  # below this OKLab chroma a source has no usable hue (black/white candies)


def _highlight_weight(L, src_L):
    """1 for specular highlights well above the candy's body tone, else 0."""
    return np.clip((L - src_L - 0.08) / 0.20, 0, 1)


def source_stats(lab, alpha, core=0.9):
    core_px = lab[alpha >= core]
    if len(core_px) == 0:
        core_px = lab[alpha > 0]
    if len(core_px) == 0:
        return None
    return np.median(core_px, axis=0)


def recolor(rgb: np.ndarray, alpha: np.ndarray, target_hex: str, *, l_gain=None):
    """Return (new_rgb, info). rgb float [0,1] HxWx3, alpha HxW in [0,1].

    In OKLab: rotate hue and scale chroma from the candy's median color to the
    target, and remap lightness around the median, so per-pixel deviations
    (shading, the printed 'm', speculars) survive the change.
    """
    lab = srgb_to_oklab(rgb)
    src = source_stats(lab, alpha)
    if src is None:
        return rgb.copy(), {"pixels": 0}
    tgt = srgb_to_oklab(parse_hex(target_hex))
    src_c, src_h = chroma_hue(src)
    tgt_c, tgt_h = chroma_hue(tgt)

    sel = alpha > 0
    px = lab[sel]
    L, a, b = px[:, 0], px[:, 1], px[:, 2]
    hw = _highlight_weight(L, src[0])

    if l_gain is None:
        # Lightening a dark candy needs its shading stretched, or it looks flat.
        l_gain = float(np.clip(tgt[0] / max(src[0], 1e-3), 0.6, 2.0))
    new_L = tgt[0] + (L - src[0]) * l_gain
    new_L = new_L * (1 - hw) + np.maximum(L, new_L) * hw

    if src_c >= ACHROMATIC_C:
        rot = tgt_h - src_h
        cos, sin = np.cos(rot), np.sin(rot)
        k = tgt_c / src_c
        new_a = (a * cos - b * sin) * k
        new_b = (a * sin + b * cos) * k
    else:
        # No hue to rotate: paint the target chroma in, fading out on highlights.
        new_a = tgt[1] * (1 - hw) + (a - src[1])
        new_b = tgt[2] * (1 - hw) + (b - src[2])

    new_lab = lab.copy()
    new_lab[sel] = np.stack([new_L, new_a, new_b], axis=-1)
    new_rgb = oklab_to_srgb(new_lab)
    w = alpha[..., None]
    out = rgb * (1 - w) + new_rgb * w
    return out, {"pixels": int((alpha >= 0.5).sum()), "source_hex": to_hex(oklab_to_srgb(src)),
                 "l_gain": round(l_gain, 3)}


def measure(rgb: np.ndarray, alpha: np.ndarray, target_hex: str, core=0.9):
    """CIEDE2000 between the median body color of the masked candies and target.

    Speculars are excluded so a glossy candy is judged on its body tone.
    """
    lab = srgb_to_oklab(rgb)
    sel = alpha >= core
    if not sel.any():
        return None
    med = np.median(lab[sel], axis=0)
    body = sel & (_highlight_weight(lab[..., 0], med[0]) < 0.1)
    body_rgb = np.median(rgb[body if body.any() else sel], axis=0)
    return {"measured_hex": to_hex(body_rgb),
            "delta_e_2000": round(float(delta_e_2000(body_rgb, parse_hex(target_hex))[0]), 2)}
