import numpy as np
import pytest

from mmrecolor.color import delta_e_2000, oklab_to_srgb, parse_hex, srgb_to_oklab, to_hex
from mmrecolor.palette import extract_palette
from mmrecolor.pipeline import run
from mmrecolor.recolor import measure, recolor
from mmrecolor.segment import segment

DOUGH = parse_hex("#E8C990")


def shaded_disk(img, cx, cy, r, hex_color):
    """Draw a candy-like disk: lit top-left, shadowed bottom-right, a specular dot."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    inside = d <= r
    base = parse_hex(hex_color)
    shade = 1.0 + 0.25 * (-(xx - cx) - (yy - cy)) / (2 * r)
    img[inside] = np.clip(base * shade[inside, None], 0, 1)
    spec = np.hypot(xx - (cx - r * 0.4), yy - (cy - r * 0.4)) <= r * 0.18
    img[spec & inside] = 0.95
    return inside


@pytest.fixture
def scene():
    img = np.ones((200, 300, 3)) * DOUGH
    reds = [shaded_disk(img, x, y, 14, "#B5221E") for x, y in [(40, 50), (120, 60), (200, 140)]]
    oranges = [shaded_disk(img, x, y, 14, "#F26A1B") for x, y in [(60, 150), (250, 50)]]
    return img, np.any(reds, axis=0), np.any(oranges, axis=0)


def test_oklab_roundtrip():
    rgb = np.random.default_rng(0).random((50, 3))
    assert np.allclose(oklab_to_srgb(srgb_to_oklab(rgb)), rgb, atol=1e-6)


def test_hex_roundtrip():
    assert to_hex(parse_hex("#1e8fd6")) == "#1E8FD6"
    assert to_hex(parse_hex("abc")) == "#AABBCC"


def test_segment_finds_red_not_orange_or_dough(scene):
    img, reds, oranges = scene
    alpha = segment(img, "red") >= 0.5
    assert (alpha & reds).sum() / reds.sum() > 0.9
    assert (alpha & oranges).sum() == 0
    assert (alpha & ~reds).sum() < 0.05 * reds.sum()


def test_recolor_hits_target_and_leaves_background(scene):
    img, reds, _ = scene
    alpha = segment(img, "red")
    out, _ = recolor(img, alpha, "#1E8FD6")
    m = measure(out, alpha, "#1E8FD6")
    assert m["delta_e_2000"] < 3.0
    untouched = alpha == 0
    assert np.allclose(out[untouched], img[untouched])


def test_recolor_keeps_shading_and_specular(scene):
    img, reds, _ = scene
    alpha = segment(img, "red")
    out, _ = recolor(img, alpha, "#1E8FD6")
    L_in = srgb_to_oklab(img)[..., 0][reds]
    L_out = srgb_to_oklab(out)[..., 0][reds]
    assert np.corrcoef(L_in, L_out)[0, 1] > 0.85
    assert out[50 - 6, 40 - 6].min() > 0.85  # specular stays white


def test_black_source_gets_colored():
    img = np.ones((80, 80, 3)) * DOUGH
    shaded_disk(img, 40, 40, 15, "#16181A")
    alpha = segment(img, "black", max_area_frac=0.5)  # tiny frame: one candy is >2%
    out, _ = recolor(img, alpha, "#FFD21F")
    assert measure(out, alpha, "#FFD21F")["delta_e_2000"] < 6.0


def test_run_report_and_exclusive_masks(scene):
    img, reds, oranges = scene
    out, alphas, report = run(img, {"red": "#1E8FD6", "orange": "#7ED321"})
    assert report["pass"], report
    assert not ((alphas["red"] >= 0.5) & (alphas["orange"] >= 0.5)).any()
    assert {r["source"] for r in report["mappings"]} == {"red", "orange"}


def test_pick_and_exclude(scene):
    img, reds, _ = scene
    _, alphas, report = run(img, {"red": "#1E8FD6"}, picks={"red": (40, 52)},
                            exclude=[(180, 120, 50, 50)])
    a = alphas["red"] >= 0.5
    assert a[50, 40] and not a[140, 200]
    assert report["mappings"][0]["mask_source"] == "pick"


def test_palette_ignores_neutral_paper():
    img = np.ones((100, 100, 3))
    img[30:70, 30:70] = parse_hex("#00A3E0")
    pal = extract_palette(img, k=3, min_chroma=0.03)
    assert delta_e_2000(parse_hex(pal[0]["hex"]), parse_hex("#00A3E0"))[0] < 2
