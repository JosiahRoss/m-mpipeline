# m-mpipeline

Drop in a shot and a color reference, get the M&M candies recolored to that
color, with a measured ΔE report.

The pixel work is deterministic, so results are exact and repeatable. AI
handles the judgment: SAM 2.1 finds the candy outlines, and Claude reads the
reference, sets the mapping, steers segmentation and checks the result.

Status: **P1**. Still images, with a classical or SAM 2.1 segmenter, plus a
Claude Code skill (`.claude/skills/mm-recolor`) that drives it.

## Install
```
pip install -e '.[dev]'           # classical segmenter + tests
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU torch (skip on GPU boxes)
pip install -e '.[sam]'           # SAM 2.1 segmenter (downloads facebook/sam2.1-hiera-small on first use)
pytest
```

## Use
```
# dominant colors of a dropped swatch/photo
mmrecolor palette swatch.jpg --min-chroma 0.03

# what candies are where (pick points come from here)
mmrecolor detect shot.png --colors red black --overlay out/detect.png

# recommended: SAM, a pick on one candy, a box around the packaging
mmrecolor recolor shot.png --map red=#1E8FD6 --pick red=628,967 \
    --segmenter sam --protect 1310,545,650,545 -o out/shot.png --debug

# classical (no torch), or a target taken from a reference image
mmrecolor recolor shot.png --map red=#1E8FD6 --pick red=628,967 \
    --exclude 1310,540,690,560 -o out/shot.png
mmrecolor recolor shot.png --ref swatch.jpg --map red=ref:1 --segmenter sam -o out/shot.png
```

| Option | Meaning |
|---|---|
| `--map SRC=TARGET` | `SRC` is one of red, orange, yellow, green, blue, brown, black. `TARGET` is a hex or `ref:N` |
| `--pick SRC=x,y` | Pixel on one typical candy. It sets the color class, and with SAM also the reference candy size |
| `--segmenter sam` | SAM 2.1 outlines (default `classical`) |
| `--protect x,y,w,h` | SAM only: box around an object to leave alone, e.g. the bag. SAM traces the object inside it |
| `--exclude x,y,w,h` | Ignore a plain box. Works with either segmenter |
| `--mask SRC=mask.png` | Use an external mask instead of segmentation |
| `--rounds N` | SAM re-prompt rounds for candies still uncovered (default 3) |
| `--tolerance` | Max CIEDE2000 for a pass (default 3.0). Exit code 2 on fail |
| `--debug` | Also write `_masks.png` and a before/after `_compare.png` |
| `--device` | `cpu` (default), `cuda` or `mps` for SAM |

Every run writes `<out>_report.json`. For each mapping it has:
- the target, measured body color, ΔE2000 and pass/fail
- `missed`: candy-shaped blobs still in the source color, for review. Some
  are real misses; others are hands or packaging correctly left alone.
- with SAM, `sam.accepted` / `sam.rejected` with the reason for each rejection

## How it works
**SAM segmenter** (`sam.py`). Color proposes, SAM outlines, and checks
decide.
1. Pixels in the candy's color window, from the pick, are split into one
   point prompt per distance-transform peak, so each candy in a pile gets
   its own prompt.
2. SAM 2.1 returns three candidate masks per point. One is kept only if
   all of these hold:
   - Its size is 0.25–4× the reference candy (from the pick). This rejects
     the mascot and the bowl contents.
   - It's convex and at most 3:1. This rejects rims and fingers. Two or
     three touching candies are allowed.
   - At least half of it is in the candy's loose color window.
   - At least 3% is solid shell color. This rejects skin lit to the same hue.
3. Inside accepted masks, only the candy's color pixels are recolored,
   with holes like the printed "m" filled. A black candy clipped by a mask
   stays black.
4. The loop re-prompts pixels not yet covered until a round adds nothing.
5. `--protect` boxes become SAM box prompts. The object is filled solid
   and never touched.

**Classical segmenter** (`segment.py`) classifies color, requires
candy-sized seeds, grows them into connected looser pixels and filters out
long thin blobs. It needs no GPU or torch, but can't tell candy from
same-hue things touching it.

**Recolor** (`recolor.py`) works in OKLab. It rotates hue and scales chroma
from the candies' median color to the target, and remaps lightness around
the median, so shading, the print and speculars survive. **Measure** is
the CIEDE2000 between the recolored body color (speculars excluded) and the
target.

### Known limits
- **Motion-blurred candies keep a thin outline of the old color.** I tried
  recoloring those blurred edge pixels, and every attempt either drew a
  light glow or skewed ΔE, so I took it back out.
- **Small candy fragments melted into cookies are often left.** They're
  below candy size or too irregular for SAM's mask to pass the checks.
  Review them in `missed`.
- **SAM on CPU takes about 80 s for a 2000 px frame**, mostly decoding the
  ~1,200 point prompts. A GPU (`--device cuda`) cuts this to seconds.
- **SAM 3**, which takes text prompts like "M&M candy", is gated on
  Hugging Face. Once you have access it can replace the color-proposed
  points.

## Roadmap
| Phase | Delivers |
|---|---|
| P0 ✅ | CLI + Claude skill: still image → recolor + ΔE report |
| P1 ✅ | SAM 2.1 segmentation, pick-sized candy checks, protect boxes, re-prompt loop, missed-candy report |
| P2 | Drag-and-drop web page, mapping confirmation chips, before/after approval |
| P3 | Video: tracking across frames (SAM 2 video), stable color over time, ProRes/EXR |
