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
Use SAM (`--segmenter sam`) whenever `python3 -c "import transformers, torch"`
works; fall back to classical otherwise (step 3b).

- `mmrecolor detect shot.png --colors red black --overlay out/detect.png`
  lists blobs per color with centers. Open the overlay.
- `--pick NAME=x,y`: a clearly visible, typical, unoccluded candy of each
  source color (a `detect` center you've confirmed is a candy; the biggest
  blob is often packaging). With SAM the pick also sets the reference candy
  size, so don't pick a tiny or half-hidden one.
- `--protect x,y,w,h`: a box around packaging (bag, box, wrapper art with
  candy icons or mascots). Packaging is out of scope unless the user says
  otherwise. Box the whole object generously; SAM traces it inside the box.

### 3b. Classical fallback
No `--protect`; use `--exclude x,y,w,h` boxes for every false hit (bag,
mascot, hands in red light). Expect more manual boxes.

## 4. Render and check
```
mmrecolor recolor shot.png --map red=#1E8FD6 --pick red=628,967 \
  --segmenter sam --protect 1310,545,650,545 -o out/shot_blue.png --debug
```
Writes `out/shot_blue.png`, `_masks.png`, `_compare.png`, `_report.json`.
Exit code 2 means a mapping missed the ΔE tolerance (default 3.0). SAM on CPU
takes ~1-2 min per 2K frame; run it in the background if needed.

Then review, yourself:
1. `_compare.png` and `_masks.png`: old-color candies left, non-candy areas
   recolored, flat-looking candies.
2. Every entry in the report's `missed` list: crop the output around each
   (x, y) and decide. Hands, skin, packaging are correct to leave. A real
   candy left behind is a miss.
3. With SAM, `sam.rejected` reasons explain misses: `size` (pick was
   atypical, re-pick), `not convex`/`elongated` (odd candy shape), `color
   share`/`no solid candy color` (heavy shadow or blur).

Fix and re-run: a better `--pick`, another `--protect`/`--exclude`, or for a
stubborn region a hand mask via `--mask red=mask.png`. Cap at ~4 rounds.

## 5. Report
Give the user the output path(s), the ΔE per mapping from the report, and an
honest list of what still isn't right (e.g. old-color outlines on
motion-blurred candies, small baked-in fragments).
