# m-mpipeline

Drop in a shot and a color reference, get the M&M candies recolored to that
color, with a measured ΔE report.

The pixel work is deterministic, so results are exact and repeatable. AI
handles the judgment: reading the reference, choosing the mapping, steering
segmentation, and checking the result.

This is **P0**: still images, CPU-only, classical segmentation, plus a Claude
Code skill (`.claude/skills/mm-recolor`) that drives it. See the
[roadmap](#roadmap) for SAM, the drop-zone UI and video.

## Install
```
pip install -e '.[dev]'
pytest
```

## Use
```
# dominant colors of a dropped swatch/photo
mmrecolor palette swatch.jpg --min-chroma 0.03

# what candies are where (pick points come from here)
mmrecolor detect shot.png --colors red black --overlay out/detect.png

# recolor red candies to a hex, or to the top color of a reference image
mmrecolor recolor shot.png --map red=#1E8FD6 --pick red=628,967 \
    --exclude 1310,540,690,560 -o out/shot.png --debug
mmrecolor recolor shot.png --ref swatch.jpg --map red=ref:1 -o out/shot.png
```

| Option | Meaning |
|---|---|
| `--map SRC=TARGET` | `SRC` is one of red, orange, yellow, green, blue, brown, black. `TARGET` is a hex or `ref:N` |
| `--pick SRC=x,y` | Pixel on one typical candy. Its color defines the class, which adapts to the shot's grade |
| `--exclude x,y,w,h` | Ignore a box (packaging, mascot, hands). Repeatable |
| `--mask SRC=mask.png` | Use an external mask (SAM, paint tool) instead of segmentation |
| `--max-area` | Drop single blobs larger than this fraction of the frame (default 0.02) |
| `--tolerance` | Max CIEDE2000 for a pass (default 3.0). Exit code 2 on fail |
| `--debug` | Also write `_masks.png` and a before/after `_compare.png` |

Every run writes `<out>_report.json`: per mapping, the target, measured body
color, ΔE2000, blob count and pass/fail.

## How it works
1. **Segment** (`segment.py`): classify pixels by OKLab hue, chroma and
   lightness. Strict seeds must contain a candy-sized disk, which rejects
   thin same-hue things like baked cookie rims. Seeds then grow into looser
   connected pixels to cover shaded and blurred edges. Long thin blobs are
   dropped, holes such as the printed "m" are filled, and the edge is
   feathered. When several colors are mapped, each pixel goes to the
   nearest hue, so masks never overlap.
2. **Recolor** (`recolor.py`): in OKLab, rotate hue and scale chroma from the
   candies' median color to the target, and remap lightness around the
   median, so shading, the print and speculars survive. Black or white
   sources have no hue, so they get the target chroma painted in, with
   lightness stretched.
3. **Measure**: CIEDE2000 between the recolored candies' median body color
   (speculars excluded) and the target.

### Known limits of P0
- Color-only segmentation can't tell candy from same-hue non-candy that
  touches it. In red-lit skin next to red candies, or a red mascot, it can
  only be fixed with `--exclude` or `--mask`. That's what SAM fixes in P1.
- Motion-blurred candies keep a faint halo of the old color.
- The ΔE checks body color only. It can't see missed candies or false hits,
  which is why the skill reviews the masks visually.

## Roadmap
| Phase | Delivers |
|---|---|
| P0 ✅ | CLI + Claude skill: still image → recolor + ΔE report |
| P1 | SAM 2/3 segmentation (`--pick` points become point prompts), automatic check-and-retry loop |
| P2 | Drag-and-drop web page, mapping confirmation chips, before/after approval |
| P3 | Video: tracking across frames, stable color over time, ProRes/EXR |
