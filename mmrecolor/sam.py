"""SAM 2.1 segmentation: color-proposed point prompts, SAM masks, candy checks.

Classical color classification only *proposes* where candies might be (one
point per distance-transform peak, so each candy in a pile gets its own
prompt). SAM draws the real boundary, and each mask must pass candy checks
(size relative to a reference candy, roundness, color share) so a red mascot,
a finger in red light or a baked cookie rim is rejected even though its
color matches.

Optional dependency: pip install -e '.[sam]'.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import srgb_to_oklab
from .segment import _fill_holes, _hue_dist, candy_radius, classify

DEFAULT_MODEL = "facebook/sam2.1-hiera-small"


class Sam2Predictor:
    """Thin wrapper over HF transformers SAM 2.1: embed once, prompt many."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu"):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()
        self.device = device

    def set_image(self, rgb: np.ndarray):
        from PIL import Image

        self.image = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
        inp = self.proc(images=self.image, return_tensors="pt")
        with self.torch.no_grad():
            self.emb = self.model.get_image_embeddings(inp["pixel_values"].to(self.device))
        self.orig = inp["original_sizes"]

    def predict(self, points, chunk: int = 16):
        """points [(x, y)] -> yields (index, masks bool [3, H, W], scores [3])."""
        for start in range(0, len(points), chunk):
            batch = points[start:start + chunk]
            inp = self.proc(images=self.image, input_points=[[[[float(x), float(y)]] for x, y in batch]],
                            input_labels=[[[1] for _ in batch]], return_tensors="pt")
            with self.torch.no_grad():
                out = self.model(input_points=inp["input_points"].to(self.device),
                                 input_labels=inp["input_labels"].to(self.device),
                                 image_embeddings=self.emb, multimask_output=True)
            low = out.pred_masks[0].cpu().numpy()  # [n, 3, 256, 256] logits
            scores = out.iou_scores[0].cpu().numpy()
            for i in range(len(batch)):
                yield start + i, self._upsample(low[i]), scores[i]

    def predict_box(self, box):
        """(x, y, w, h) -> bool mask of the object SAM finds in that box."""
        x, y, w, h = box
        inp = self.proc(images=self.image, input_boxes=[[[float(x), float(y), float(x + w), float(y + h)]]],
                        return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(input_boxes=inp["input_boxes"].to(self.device),
                             image_embeddings=self.emb, multimask_output=False)
        return self._upsample(out.pred_masks[0, 0].cpu().numpy())[0]

    def _upsample(self, logits):
        """Low-res logits -> bool masks at image size. SAM 2 resizes the image
        straight to a square (no padding), so resize straight back."""
        h, w = int(self.orig[0][0]), int(self.orig[0][1])
        return np.stack([cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR) > 0
                         for lg in logits])


@dataclass
class SamResult:
    alpha: np.ndarray
    accepted: list = field(default_factory=list)   # [{x, y, area, score}]
    rejected: list = field(default_factory=list)   # [{x, y, reason}]
    ref_area: float = 0.0


def propose_points(candidates: np.ndarray, radius: float, covered=None, limit=400):
    """One prompt per candy: peaks of the distance transform of candidate pixels."""
    region = candidates.copy()
    if covered is not None:
        region &= ~covered
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    r = max(2, int(round(radius)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    peaks = (dist >= max(1.5, radius * 0.5)) & (dist >= cv2.dilate(dist, k) - 1e-6)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-dist[ys, xs])
    taken, pts = np.zeros_like(region), []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if taken[y, x]:
            continue
        pts.append((x, y))
        cv2.circle(taken.view(np.uint8), (x, y), int(radius * 1.5), 1, -1)
        if len(pts) >= limit:
            break
    return pts


def _shape_ok(mask: np.ndarray, min_solidity: float = 0.85):
    """(ok, reason). Candies are convex blobs, at most ~3:1 (tilted or motion blurred)."""
    u8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "empty"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull <= 0 or area / hull < min_solidity:
        return False, f"not convex (solidity {area / max(hull, 1):.2f})"
    (_, _), (a, b), _ = cv2.minAreaRect(c)
    if min(a, b) <= 0 or max(a, b) / min(a, b) > 3.0:
        return False, "elongated"
    return True, ""


def protect_mask(predictor, boxes, dilate=4):
    """Masks of whole objects (packaging, a mascot) to leave untouched.

    Each is a box prompt: unlike a single click, which SAM may read as just
    the logo or the lettering, a box names the whole object. The result is
    clipped to the box so a loose mask can't spill over the scene.
    """
    out = None
    for x, y, bw, bh in boxes:
        m = predictor.predict_box((x, y, bw, bh))
        clip = np.zeros_like(m)
        clip[y:y + bh, x:x + bw] = True
        m = (m & clip).astype(np.uint8)
        # A protected object is protected whole: fill its holes (SAM can leave
        # gaps at printed details like a mascot's face).
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(m, cnts, -1, 1, thickness=-1)
        m = m.astype(bool)
        out = m if out is None else out | m
    if out is not None and dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        out = cv2.dilate(out.astype(np.uint8), k).astype(bool)
    return out


def sam_segment(rgb, rule, predictor, *, own=None, exclude=(), blocked=None, pick=None,
                rounds=3, feather=1.5, min_color_share=0.5,
                min_core_share=0.03) -> SamResult:
    """Mask candies of one color class with SAM, re-prompting on misses.

    rule: the class's ClassRule (strict); its loose() window proposes points.
    own: bool mask restricting to pixels whose hue is nearest this class.
    blocked: bool mask never to touch (from protect_mask).
    pick: (x, y) on a known candy; its SAM mask sets the reference size.
    """
    lab = srgb_to_oklab(rgb)
    loose = classify(lab, rule.loose())
    if own is not None:
        loose &= own
    for x, y, w, h in exclude:
        loose[y:y + h, x:x + w] = False
    if blocked is not None:
        loose &= ~blocked
    strict = classify(lab, rule)
    radius = candy_radius(rgb.shape)

    res = SamResult(alpha=np.zeros(rgb.shape[:2], np.float32))
    union = np.zeros(rgb.shape[:2], bool)
    tried = np.zeros(rgb.shape[:2], bool)

    if pick is not None:
        _, masks, scores = next(predictor.predict([pick]))
        ok = [i for i in range(len(scores)) if masks[i].sum() > 0 and _shape_ok(masks[i])[0]]
        best = min(ok or [int(np.argmax(scores))], key=lambda i: masks[i].sum())
        res.ref_area = float(masks[best].sum())

    for _ in range(rounds):
        # Each round prompts candidate pixels not yet covered or tried; a
        # round that accepts nothing new ends the loop.
        pts = propose_points(loose, radius, covered=union | tried)
        if not pts:
            break
        found = []
        for i, masks, scores in predictor.predict(pts):
            found.append((pts[i], masks, scores))
        if not res.ref_area:
            # No pick: the typical smallest confident mask is a single candy.
            areas = [m[0].sum() for _, m, s in found if s[0] > 0.5 and m[0].sum() > 0]
            res.ref_area = float(np.median(areas)) if areas else np.pi * (3 * radius) ** 2
        lo, hi = 0.25 * res.ref_area, 4.0 * res.ref_area
        new = 0
        for (x, y), masks, scores in found:
            cv2.circle(tried.view(np.uint8), (x, y), int(radius), 1, -1)
            choice, reason = None, "no candy-sized mask"
            for j in np.argsort(-scores):
                m = masks[j]
                area = m.sum()
                if not lo <= area <= hi:
                    reason = f"size {int(area)} vs ref {int(res.ref_area)}"
                    continue
                # Two or three touching candies make a non-convex mask; that is
                # fine at that size, since the color gate below keeps a black
                # neighbour inside the mask untouched.
                ok, why = _shape_ok(m, 0.6 if area <= 2.5 * res.ref_area else 0.85)
                if not ok:
                    reason = why
                    continue
                share = loose[m].mean() if area else 0
                if share < min_color_share:
                    reason = f"color share {share:.2f}"
                    continue
                # Every real candy has a core of solid shell color; skin or
                # wood in the same light only reaches the loose window.
                core = strict[m].mean()
                if core < min_core_share:
                    reason = f"no solid candy color (core {core:.2f})"
                    continue
                choice = j
                break
            if choice is None:
                res.rejected.append({"x": x, "y": y, "reason": reason})
                continue
            m = masks[choice]
            if (m & union).sum() > 0.8 * m.sum():
                continue
            union |= m
            new += 1
            res.accepted.append({"x": x, "y": y, "area": int(m.sum()),
                                 "score": round(float(scores[choice]), 3)})
        if new == 0:
            break

    # SAM is the shape gate, color the pixel gate: inside accepted masks only
    # recolor this class's pixels (a mask that clips a neighboring black candy
    # must not tint it), then fill enclosed holes (print, speculars). SAM
    # boundaries also sit inside motion blur, so take the same-color halo.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    near = cv2.dilate(union.astype(np.uint8), k3, iterations=3).astype(bool)
    binary = _fill_holes(near & loose)
    alpha = binary.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha[~cv2.dilate(binary.astype(np.uint8), k3, iterations=2).astype(bool)] = 0
    res.alpha = np.clip(alpha, 0, 1)
    return res


def nearest_hue_masks(rgb, rules):
    """{name: bool} pixel ownership by nearest hue among chromatic classes."""
    lab = srgb_to_oklab(rgb)
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    chromatic = [n for n, r in rules.items() if not r.achromatic]
    if len(chromatic) < 2:
        return {}
    nearest = np.argmin(np.stack([_hue_dist(hue, rules[n].hue) for n in chromatic]), axis=0)
    return {n: nearest == i for i, n in enumerate(chromatic)}
