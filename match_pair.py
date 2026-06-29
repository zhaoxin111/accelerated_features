"""
Run XFeat matching on two image files and save an OpenCV match visualization.

The defaults target the sample pair under data/pair1 so the script can be
used immediately after cloning or placing images in that folder.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from modules.xfeat import XFeat


def parse_args():
    parser = argparse.ArgumentParser(description="Match two images with XFeat and save a visualization.")
    parser.add_argument("--image0", default="data/pair1/1776299438444_prev.jpg", help="Path to the first image.")
    parser.add_argument("--image1", default="data/pair1/1776299438444_curr.jpg", help="Path to the second image.")
    parser.add_argument("--output", default="outputs/pair1_xfeat_matches.jpg", help="Output visualization path.")
    parser.add_argument("--top-k", type=int, default=4096, help="Maximum number of XFeat keypoints per image.")
    parser.add_argument("--min-cossim", type=float, default=0.82, help="Minimum cosine similarity for mutual matches.")
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
        "--upward-top-fraction",
        type=float,
        default=0.3,
        help="Visualize the top fraction of upward-dominant matches by upward displacement.",
    )
    parser.add_argument(
        "--no-ransac",
        action="store_true",
        help="Draw all descriptor matches instead of homography inliers.",
    )
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def homography_inlier_mask(points0, points1, ransac_thr):
    """Return a boolean inlier mask when enough matches support a homography."""
    if len(points0) < 4:
        return None

    _, mask = cv2.findHomography(
        points0.astype(np.float32),
        points1.astype(np.float32),
        cv2.USAC_MAGSAC,
        ransac_thr,
        maxIters=1000,
        confidence=0.995,
    )
    if mask is None:
        return None
    return mask.ravel().astype(bool)


def select_match_indices(points0, inlier_mask=None, max_draw_matches=200, sample_seed=0):
    """Select the filtered and sampled match indices used by all visualizations."""
    if inlier_mask is None:
        selected = np.ones(len(points0), dtype=bool)
    else:
        selected = inlier_mask

    selected_indices = np.flatnonzero(selected)
    if max_draw_matches > 0 and len(selected_indices) > max_draw_matches:
        rng = np.random.default_rng(sample_seed)
        selected_indices = np.sort(rng.choice(selected_indices, size=max_draw_matches, replace=False))
    return selected_indices


def select_upward_dominant_indices(
    points0,
    points1,
    inlier_mask=None,
    max_draw_matches=200,
    sample_seed=0,
    upward_min_pixels=1.0,
    upward_dominance_ratio=1.0,
    upward_top_fraction=0.3,
):
    """Select strongest upward-dominant matches by image1 upward displacement."""
    dx = points1[:, 0] - points0[:, 0]
    dy = points1[:, 1] - points0[:, 1]
    upward = dy <= -upward_min_pixels
    vertical_dominant = np.abs(dy) >= upward_dominance_ratio * np.abs(dx)
    selected = upward & vertical_dominant
    if inlier_mask is not None:
        selected &= inlier_mask

    candidate_indices = np.flatnonzero(selected)
    top_fraction = np.clip(upward_top_fraction, 0.0, 1.0)
    if len(candidate_indices) > 0 and top_fraction > 0:
        top_count = max(1, int(np.ceil(len(candidate_indices) * top_fraction)))
        upward_displacement = -dy[candidate_indices]
        top_order = np.argsort(-upward_displacement)[:top_count]
        top_indices = np.sort(candidate_indices[top_order])
    else:
        top_indices = np.array([], dtype=np.int64)

    draw_indices = top_indices
    if max_draw_matches > 0 and len(draw_indices) > max_draw_matches:
        rng = np.random.default_rng(sample_seed)
        draw_indices = np.sort(rng.choice(draw_indices, size=max_draw_matches, replace=False))
    return candidate_indices, top_indices, draw_indices


def put_label(image, text, origin=(12, 32), font_scale=0.9, thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def pad_to_height(image, height):
    """Pad an image at the bottom so side-by-side panels share one height."""
    if image.shape[0] == height:
        return image
    padded = np.zeros((height, image.shape[1], 3), dtype=image.dtype)
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def draw_point_panel(image, points, label, color, point_radius):
    """Draw matched point locations on a single image panel."""
    panel = image.copy()
    outline_radius = max(point_radius + 1, 1)
    for point in points:
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(panel, center, outline_radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, center, point_radius, color, -1, cv2.LINE_AA)
    put_label(panel, label, origin=(8, 22), font_scale=0.55, thickness=1)
    return panel


def draw_matches(
    image0,
    image1,
    points0,
    points1,
    selected_indices,
    inlier_mask=None,
    title_scale=0.45,
    label=None,
    match_color=(0, 220, 0),
):
    """Convert sampled point arrays to OpenCV keypoints and draw matches."""
    keypoints0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in points0[selected_indices]]
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in points1[selected_indices]]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]

    canvas = cv2.drawMatches(
        image0,
        keypoints0,
        image1,
        keypoints1,
        matches,
        None,
        matchColor=match_color,
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if label is None:
        label = f"XFeat matches: {len(points0)}"
        if inlier_mask is not None:
            label += f" | RANSAC inliers: {int(inlier_mask.sum())}"
        label += f" | drawn: {len(selected_indices)}"
    put_label(canvas, label, origin=(8, 22), font_scale=title_scale, thickness=1)
    return canvas


def draw_visualization(
    image0,
    image1,
    points0,
    points1,
    selected_indices,
    upward_draw_indices,
    inlier_mask=None,
    point_radius=2,
    title_scale=0.45,
):
    """Create match, sampled-point, and upward-dominant point visualizations."""
    match_canvas = draw_matches(
        image0,
        image1,
        points0,
        points1,
        selected_indices,
        inlier_mask=inlier_mask,
        title_scale=title_scale,
    )

    point_panel0 = draw_point_panel(
        image0,
        points0[selected_indices],
        f"image0 pts: {len(selected_indices)}",
        (0, 220, 0),
        point_radius,
    )
    point_panel1 = draw_point_panel(
        image1,
        points1[selected_indices],
        f"image1 pts: {len(selected_indices)}",
        (255, 0, 255),
        point_radius,
    )
    panel_height = max(point_panel0.shape[0], point_panel1.shape[0])
    point_canvas = np.hstack((pad_to_height(point_panel0, panel_height), pad_to_height(point_panel1, panel_height)))

    canvas_width = max(match_canvas.shape[1], point_canvas.shape[1])
    match_canvas = pad_to_width(match_canvas, canvas_width)
    point_canvas = pad_to_width(point_canvas, canvas_width)

    upward_panel = draw_point_panel(
        image1,
        points1[upward_draw_indices],
        f"up pts: {len(upward_draw_indices)}",
        (255, 255, 0),
        point_radius,
    )
    upward_canvas = pad_to_width(upward_panel, canvas_width)

    upward_match_canvas = draw_matches(
        image0,
        image1,
        points0,
        points1,
        upward_draw_indices,
        title_scale=title_scale,
        label=f"final upward matches: {len(upward_draw_indices)}",
        match_color=(255, 255, 0),
    )
    upward_match_canvas = pad_to_width(upward_match_canvas, canvas_width)
    return np.vstack((match_canvas, point_canvas, upward_canvas, upward_match_canvas))


def pad_to_width(image, width):
    """Pad an image at the right so stacked rows share one width."""
    if image.shape[1] == width:
        return image
    padded = np.zeros((image.shape[0], width, 3), dtype=image.dtype)
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def main():
    args = parse_args()
    image0_path = Path(args.image0)
    image1_path = Path(args.image1)
    output_path = Path(args.output)

    image0 = read_image(image0_path)
    image1 = read_image(image1_path)

    matcher = XFeat(top_k=args.top_k)
    points0, points1 = matcher.match_xfeat(image0, image1, top_k=args.top_k, min_cossim=args.min_cossim)

    inlier_mask = None
    if not args.no_ransac:
        inlier_mask = homography_inlier_mask(points0, points1, args.ransac_thr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = select_match_indices(
        points0,
        inlier_mask=inlier_mask,
        max_draw_matches=args.max_draw_matches,
        sample_seed=args.sample_seed,
    )
    upward_candidates, upward_top_indices, upward_draw_indices = select_upward_dominant_indices(
        points0,
        points1,
        inlier_mask=inlier_mask,
        max_draw_matches=args.max_draw_matches,
        sample_seed=args.sample_seed + 1,
        upward_min_pixels=args.upward_min_pixels,
        upward_dominance_ratio=args.upward_dominance_ratio,
        upward_top_fraction=args.upward_top_fraction,
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
    )
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print(f"image0: {image0_path}")
    print(f"image1: {image1_path}")
    print(f"matches: {len(points0)}")
    if inlier_mask is not None:
        print(f"ransac_inliers: {int(inlier_mask.sum())}")
    print(f"drawn_matches: {len(selected_indices)}")
    print(f"upward_dominant_candidates: {len(upward_candidates)}")
    print(f"upward_dominant_top_matches: {len(upward_top_indices)}")
    print(f"upward_dominant_drawn: {len(upward_draw_indices)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
