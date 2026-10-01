"""Command line: `mmrecolor palette` and `mmrecolor recolor`."""

import argparse
import json
import sys
from pathlib import Path

from .color import parse_hex, to_hex
from .palette import extract_palette
from .pipeline import load_rgb, overlay, run, save_rgb, side_by_side
from .segment import CANDY_COLORS, default_rule, segment_many


def _box(s):
    x, y, w, h = (int(v) for v in s.split(","))
    return (x, y, w, h)


def _point(s):
    k, v = _pair(s)
    x, y = (int(t) for t in v.split(","))
    return k, (x, y)


def _pair(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip().lower(), v.strip()


def _resolve_target(value, ref_palettes):
    """'#00A3E0' | '00A3E0' | 'ref:1' (1-based color from --ref palettes)."""
    if value.lower().startswith("ref:"):
        idx = int(value[4:]) - 1
        if not 0 <= idx < len(ref_palettes):
            raise SystemExit(f"{value}: --ref gave only {len(ref_palettes)} colors")
        return ref_palettes[idx]
    return to_hex(parse_hex(value))


def cmd_palette(args):
    for path in args.images:
        pal = extract_palette(load_rgb(path), k=args.k, min_chroma=args.min_chroma)
        print(json.dumps({"image": path, "palette": pal}, indent=2))


def cmd_detect(args):
    """List candidate candy blobs per color so a pick point can be chosen from data."""
    import cv2
    import numpy as np

    image = load_rgb(args.image)
    names = args.colors or list(CANDY_COLORS)
    alphas = segment_many(image, {n: default_rule(n) for n in names},
                          exclude=args.exclude or [], max_area_frac=args.max_area)
    result = {}
    for name, alpha in alphas.items():
        n, _, stats, cents = cv2.connectedComponentsWithStats((alpha >= 0.5).astype(np.uint8))
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:args.top] + 1
        result[name] = [{"x": int(cents[i][0]), "y": int(cents[i][1]),
                         "area": int(stats[i, cv2.CC_STAT_AREA])} for i in order]
    if args.overlay:
        save_rgb(args.overlay, overlay(image, alphas))
    print(json.dumps(result, indent=2))


def cmd_recolor(args):
    image = load_rgb(args.image)
    ref_colors = []
    for path in args.ref or []:
        pal = extract_palette(load_rgb(path), k=args.ref_k, min_chroma=0.03)
        if pal:
            ref_colors.append(pal[0]["hex"])
    mapping = {}
    for name, value in args.map:
        if name not in CANDY_COLORS:
            raise SystemExit(f"unknown candy color {name!r}; choose from {', '.join(CANDY_COLORS)}")
        mapping[name] = _resolve_target(value, ref_colors)
    masks = dict(args.mask or [])
    picks = dict(args.pick or [])

    out, alphas, report = run(image, mapping, masks=masks, picks=picks, exclude=args.exclude or [],
                              max_area_frac=args.max_area, tolerance=args.tolerance)
    report["image"] = args.image
    report["ref_colors"] = ref_colors

    out_path = Path(args.out)
    save_rgb(out_path, out)
    stem = out_path.with_suffix("")
    if args.debug:
        save_rgb(f"{stem}_masks.png", overlay(image, alphas))
        save_rgb(f"{stem}_compare.png", side_by_side(image, out))
    Path(f"{stem}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


def main(argv=None):
    p = argparse.ArgumentParser(prog="mmrecolor", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("palette", help="dominant colors of reference images")
    pp.add_argument("images", nargs="+")
    pp.add_argument("-k", type=int, default=5)
    pp.add_argument("--min-chroma", type=float, default=0.0)
    pp.set_defaults(fn=cmd_palette)

    pd = sub.add_parser("detect", help="list candidate candy blobs per color")
    pd.add_argument("image")
    pd.add_argument("--colors", nargs="+", choices=list(CANDY_COLORS))
    pd.add_argument("--exclude", type=_box, action="append")
    pd.add_argument("--max-area", type=float, default=0.02)
    pd.add_argument("--top", type=int, default=10)
    pd.add_argument("--overlay", help="write a mask overlay PNG here")
    pd.set_defaults(fn=cmd_detect)

    pr = sub.add_parser("recolor", help="recolor candies in a still image")
    pr.add_argument("image")
    pr.add_argument("--map", type=_pair, action="append", required=True,
                    help="SOURCE=TARGET, e.g. red=#00A3E0 or orange=ref:1")
    pr.add_argument("--ref", action="append", help="color reference image (repeatable; ref:N is the Nth)")
    pr.add_argument("--ref-k", type=int, default=4)
    pr.add_argument("--mask", type=_pair, action="append",
                    help="SOURCE=mask.png to override classical segmentation")
    pr.add_argument("--pick", type=_point, action="append",
                    help="SOURCE=x,y pixel on one candy of that color; adapts to the shot's grade")
    pr.add_argument("--exclude", type=_box, action="append", help="x,y,w,h box to ignore (repeatable)")
    pr.add_argument("--max-area", type=float, default=0.02,
                    help="drop blobs larger than this fraction of the frame")
    pr.add_argument("--tolerance", type=float, default=3.0, help="max CIEDE2000 to pass")
    pr.add_argument("-o", "--out", required=True)
    pr.add_argument("--debug", action="store_true", help="also write _masks.png and _compare.png")
    pr.set_defaults(fn=cmd_recolor)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
