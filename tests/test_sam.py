"""SAM path logic with a fake predictor (no model download needed).

The fake returns, for a clicked pixel, the connected region of similar color
as its first mask and progressively larger regions as the other two, which
is roughly how SAM's multimask output behaves on flat-colored objects.
"""

import cv2
import numpy as np

from mmrecolor.color import parse_hex
from mmrecolor.pipeline import run
from mmrecolor.sam import propose_points

from test_pipeline import DOUGH, shaded_disk


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def set_image(self, rgb):
        self.rgb = rgb

    def predict(self, points, chunk=16):
        h, w = self.rgb.shape[:2]
        for i, (x, y) in enumerate(points):
            self.calls += 1
            seed = self.rgb[y, x]
            near = np.linalg.norm(self.rgb - seed, axis=-1) < 0.25
            _, lab = cv2.connectedComponents(near.astype(np.uint8))
            m0 = lab == lab[y, x] if near[y, x] else np.zeros((h, w), bool)
            k = np.ones((15, 15), np.uint8)
            m1 = cv2.dilate(m0.astype(np.uint8), k).astype(bool)
            m2 = np.ones((h, w), bool)
            yield i, np.stack([m0, m1, m2]), np.array([0.9, 0.6, 0.3])

    def predict_box(self, box):
        x, y, w, h = box
        m = np.zeros(self.rgb.shape[:2], bool)
        m[y:y + h, x:x + w] = True
        return m


def scene():
    img = np.ones((240, 360, 3)) * DOUGH
    candies = [(40, 50), (110, 50), (180, 60), (60, 170)]
    for x, y in candies:
        shaded_disk(img, x, y, 13, "#B5221E")
    shaded_disk(img, 290, 150, 55, "#B5221E")       # mascot: same red, far bigger
    img[200:212, 120:250] = parse_hex("#B5221E")     # long red bar (a rim/sleeve)
    shaded_disk(img, 160, 170, 13, "#16181A")        # black candy next to reds
    return img, candies


def test_sam_keeps_candies_rejects_mascot_and_bar():
    img, candies = scene()
    out, alphas, report = run(img, {"red": "#1E8FD6"}, picks={"red": (40, 50)},
                              segmenter="sam", predictor=FakePredictor())
    a = alphas["red"] >= 0.5
    for x, y in candies:
        assert a[y, x], (x, y)
    assert not a[150, 290], "mascot recolored"
    assert not a[206, 180], "bar recolored"
    assert not a[170, 160], "black candy tinted"
    row = report["mappings"][0]
    assert row["pass"] and row["sam"]["accepted"] >= len(candies)
    reasons = " ".join(r["reason"] for r in row["sam"]["rejected"])
    assert "size" in reasons or "elongated" in reasons
    # The mascot is still red, so QA reports it for a human/Claude to judge.
    assert any(abs(m["x"] - 290) < 20 for m in row["missed"])


def test_protect_point_blocks_object():
    img, _ = scene()
    _, alphas, report = run(img, {"red": "#1E8FD6"}, picks={"red": (40, 50)},
                            protect=[(230, 90, 120, 120)], segmenter="sam", predictor=FakePredictor())
    assert alphas["protected"][150, 290] == 1
    assert not any(abs(m["x"] - 290) < 20 for m in report["mappings"][0]["missed"])


def test_one_prompt_per_candy_in_a_pile():
    m = np.zeros((100, 200), bool)
    for cx in (40, 66, 92, 118):                    # four touching candies
        cv2.circle(m.view(np.uint8), (cx, 50), 13, 1, -1)
    pts = propose_points(m, radius=8)
    assert 3 <= len(pts) <= 6
