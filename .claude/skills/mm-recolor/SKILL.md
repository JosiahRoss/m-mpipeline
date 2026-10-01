---
name: mm-recolor
description: Recolor M&M candies in a still image to match a color reference (hex, Pantone value, swatch or photo). Use when someone drops a shot plus a color ref and wants the candies changed to that color.
---

# M&M recolor

You drive the `mmrecolor` CLI in this repo. The pixel work is deterministic;
your job is judgment: read the inputs, choose the mapping, steer the
segmenter, and check the result with your own eyes plus the ΔE report.

Run commands from the repo root (`python3 -m mmrecolor ...`, or `mmrecolor`
after `pip install -e .`). Write outputs to `out/` (git-ignored).

## 1. Read the inputs
- Look at the shot. Note which candy colors are actually present (graded
  footage drifts: Halloween "orange" can read as red) and any same-colored
  non-candy things: packaging, mascots, pumpkins, skin, baked cookie rims.
- Color reference: a hex is used as-is. For an image, run
  `mmrecolor palette ref.png --min-chroma 0.03` and take the top chromatic
  color, or pass it straight through with `--ref ref.png --map red=ref:1`.
  A Pantone name needs its hex from the client spec; don't guess one.

## 2. Confirm the mapping
State it plainly, e.g. "red → #1E8FD6, black stays", and ask the user to
confirm before rendering if more than one candy color is in play or the
request is ambiguous about which candies change.

## 3. Find the candies
- `mmrecolor detect shot.png --colors red black --overlay out/detect.png`
  lists blobs per color with centers. Open the overlay.
- Choose a `--pick NAME=x,y` on a clearly visible, typical candy of each
  source color (a center from `detect` that you've confirmed is a candy, not
  the biggest blob, which is often packaging). Picks adapt to the shot's grade.
- Add `--exclude x,y,w,h` boxes for false hits (bag art, mascot, hands).
  Packaging is out of scope unless the user says otherwise.

## 4. Render and check
```
mmrecolor recolor shot.png --map red=#1E8FD6 --pick red=628,967 \
  --exclude 1310,540,690,560 -o out/shot_blue.png --debug
```
Writes `out/shot_blue.png`, `_masks.png`, `_compare.png`, `_report.json`.
Exit code 2 means a mapping missed the ΔE tolerance (default 3.0).

Then look at `_compare.png` and `_masks.png` yourself and check for:
- candies left in the old color (missed, or a rim of old color around edges)
- non-candy areas recolored (hands, cookie rims, pumpkin, bag)
- flat-looking candies (lost shading / highlights)

Fix with another `--exclude`, a better `--pick`, or `--max-area` (raise it
if a pile of touching candies was dropped), and re-run. Cap at ~4 rounds;
if classical segmentation can't separate something (e.g. skin in red light
touching red candies), say so and suggest a mask from SAM or a paint tool via
`--mask red=mask.png`.

## 5. Report
Give the user the output path(s), the ΔE per mapping from the report, and an
honest list of what still isn't right.
