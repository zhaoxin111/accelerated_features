"""
Run XFeat matching for every image pair under a data directory.

Each immediate subdirectory is treated as one pair. The script prefers
*_prev.* -> *_curr.* ordering, and falls back to sorted image files when those
names are not present.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2

from match_pair import (
    XFeat,
    draw_visualization,
    homography_inlier_mask,
    match_images,
    matcher_label,
    matcher_slug,
    read_image,
    select_match_indices,
    select_upward_dominant_indices,
    validate_matcher_args,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run XFeat matching on all pairs under data/.")
    parser.add_argument("--data-dir", default="data", help="Directory containing pair subdirectories.")
    parser.add_argument("--output-dir", default="outputs/all_pairs", help="Directory for visualizations and CSV summary.")
    parser.add_argument("--top-k", type=int, default=4096, help="Maximum number of XFeat keypoints per image.")
    parser.add_argument("--min-cossim", type=float, default=0.82, help="Minimum cosine similarity for sparse mutual matches.")
    parser.add_argument(
        "--semi-dense",
        action="store_true",
        help="Use semi-dense XFeat* matching instead of sparse XFeat matching.",
    )
    parser.add_argument(
        "--lighterglue",
        action="store_true",
        help="Use XFeat sparse features matched by LighterGlue.",
    )
    parser.add_argument(
        "--lighterglue-min-conf",
        type=float,
        default=0.1,
        help="Minimum LighterGlue match confidence.",
    )
    parser.add_argument("--ransac-thr", type=float, default=4.0, help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--max-draw-matches", type=int, default=200, help="Maximum number of matches to draw.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Random seed used when sampling drawn matches.")
    parser.add_argument("--point-radius", type=int, default=2, help="Radius of matched-point circles in the subplot.")
    parser.add_argument("--title-scale", type=float, default=0.45, help="Font scale of the top match summary title.")
    parser.add_argument("--upward-min-pixels", type=float, default=1.0, help="Minimum upward y displacement in image1.")
    parser.add_argument(
        "--upward-dominance-ratio",
        type=float,
        default=1.0,
        help="Require abs(dy) to be at least this ratio times abs(dx).",
    )
    parser.add_argument(
        "--upward-top-k",
        type=int,
        default=30,
        help="Visualize this many upward-dominant matches with the strongest upward displacement. Use 0 for all.",
    )
    parser.add_argument(
        "--no-ransac",
        action="store_true",
        help="Draw all descriptor matches instead of homography inliers.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue processing remaining pairs when one pair fails.",
    )
    args = parser.parse_args()
    validate_matcher_args(parser, args)
    return args


def image_files(pair_dir):
    return sorted(path for path in pair_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def find_pair_images(pair_dir):
    files = image_files(pair_dir)
    if len(files) < 2:
        return None

    prev_files = [path for path in files if "prev" in path.stem.lower()]
    curr_files = [path for path in files if "curr" in path.stem.lower()]
    if prev_files and curr_files:
        return prev_files[0], curr_files[0]

    return files[0], files[1]


def iter_pair_dirs(data_dir):
    return sorted(path for path in data_dir.iterdir() if path.is_dir())


def process_pair(pair_dir, image0_path, image1_path, output_path, matcher, args, pair_index):
    image0 = read_image(image0_path)
    image1 = read_image(image1_path)
    points0, points1 = match_images(
        matcher,
        image0,
        image1,
        top_k=args.top_k,
        min_cossim=args.min_cossim,
        semi_dense=args.semi_dense,
        lighterglue=args.lighterglue,
        lighterglue_min_conf=args.lighterglue_min_conf,
    )

    inlier_mask = None
    if not args.no_ransac:
        inlier_mask = homography_inlier_mask(points0, points1, args.ransac_thr)

    selected_indices = select_match_indices(
        points0,
        inlier_mask=inlier_mask,
        max_draw_matches=args.max_draw_matches,
        sample_seed=args.sample_seed + pair_index,
    )
    upward_candidates, upward_top_indices, upward_draw_indices = select_upward_dominant_indices(
        points0,
        points1,
        inlier_mask=inlier_mask,
        max_draw_matches=args.max_draw_matches,
        sample_seed=args.sample_seed + pair_index + 10000,
        upward_min_pixels=args.upward_min_pixels,
        upward_dominance_ratio=args.upward_dominance_ratio,
        upward_top_k=args.upward_top_k,
    )

    canvas = draw_visualization(
        image0,
        image1,
        points0,
        points1,
        selected_indices,
        upward_draw_indices,
        inlier_mask=inlier_mask,
        point_radius=args.point_radius,
        title_scale=args.title_scale,
        semi_dense=args.semi_dense,
        lighterglue=args.lighterglue,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write output image: {output_path}")

    return {
        "pair": pair_dir.name,
        "image0": str(image0_path),
        "image1": str(image1_path),
        "matcher": matcher_label(semi_dense=args.semi_dense, lighterglue=args.lighterglue),
        "matches": len(points0),
        "ransac_inliers": int(inlier_mask.sum()) if inlier_mask is not None else "",
        "drawn_matches": len(selected_indices),
        "upward_dominant_candidates": len(upward_candidates),
        "upward_dominant_top_matches": len(upward_top_indices),
        "upward_dominant_drawn": len(upward_draw_indices),
        "output": str(output_path),
        "status": "ok",
        "error": "",
    }


def write_summary(summary_path, rows):
    fieldnames = [
        "pair",
        "image0",
        "image1",
        "matcher",
        "matches",
        "ransac_inliers",
        "drawn_matches",
        "upward_dominant_candidates",
        "upward_dominant_top_matches",
        "upward_dominant_drawn",
        "output",
        "status",
        "error",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {data_dir}")

    matcher = XFeat(top_k=args.top_k)
    rows = []

    for pair_index, pair_dir in enumerate(iter_pair_dirs(data_dir)):
        pair_images = find_pair_images(pair_dir)
        if pair_images is None:
            row = {
                "pair": pair_dir.name,
                "image0": "",
                "image1": "",
                "matcher": matcher_label(semi_dense=args.semi_dense, lighterglue=args.lighterglue),
                "matches": "",
                "ransac_inliers": "",
                "drawn_matches": "",
                "upward_dominant_candidates": "",
                "upward_dominant_top_matches": "",
                "upward_dominant_drawn": "",
                "output": "",
                "status": "skipped",
                "error": "fewer than two image files",
            }
            rows.append(row)
            print(f"[skip] {pair_dir.name}: fewer than two image files")
            continue

        image0_path, image1_path = pair_images
        mode_slug = matcher_slug(semi_dense=args.semi_dense, lighterglue=args.lighterglue)
        output_path = output_dir / f"{pair_dir.name}_{mode_slug}.jpg"
        try:
            row = process_pair(pair_dir, image0_path, image1_path, output_path, matcher, args, pair_index)
            rows.append(row)
            print(
                f"[ok] {pair_dir.name}: matches={row['matches']} "
                f"upward={row['upward_dominant_drawn']} output={output_path}"
            )
        except Exception as exc:
            if not args.skip_errors:
                raise
            row = {
                "pair": pair_dir.name,
                "image0": str(image0_path),
                "image1": str(image1_path),
                "matcher": matcher_label(semi_dense=args.semi_dense, lighterglue=args.lighterglue),
                "matches": "",
                "ransac_inliers": "",
                "drawn_matches": "",
                "upward_dominant_candidates": "",
                "upward_dominant_top_matches": "",
                "upward_dominant_drawn": "",
                "output": str(output_path),
                "status": "error",
                "error": str(exc),
            }
            rows.append(row)
            print(f"[error] {pair_dir.name}: {exc}")

    mode_slug = matcher_slug(semi_dense=args.semi_dense, lighterglue=args.lighterglue)
    summary_path = output_dir / f"summary_{mode_slug}.csv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
