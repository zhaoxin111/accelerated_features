"""
Small external-facing API for XFeat keypoint detection and image matching.

This module is intentionally independent from the demo scripts so other repos
can import one stable class without depending on visualization code.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

from modules.xfeat import XFeat


@dataclass
class MatchResult:
    points0: np.ndarray
    points1: np.ndarray
    matches: np.ndarray
    inlier_mask: Optional[np.ndarray]
    method: str

    @property
    def num_matches(self) -> int:
        return int(len(self.points0))

    @property
    def num_inliers(self) -> Optional[int]:
        if self.inlier_mask is None:
            return None
        return int(self.inlier_mask.sum())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
            "points0": self.points0.tolist(),
            "points1": self.points1.tolist(),
            "matches": self.matches.tolist(),
            "inlier_mask": None if self.inlier_mask is None else self.inlier_mask.astype(bool).tolist(),
        }


class XFeatMatchingAPI:
    """Reusable detector and matcher wrapper for external repos."""

    def __init__(
        self,
        weights: Optional[str] = None,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
    ) -> None:
        self.top_k = top_k
        if weights is None:
            self.matcher = XFeat(top_k=top_k, detection_threshold=detection_threshold)
        else:
            self.matcher = XFeat(weights=weights, top_k=top_k, detection_threshold=detection_threshold)

    def detect(self, image: np.ndarray, top_k: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return keypoints, descriptors, and scores as numpy arrays."""
        result = self.matcher.detectAndCompute(normalize_image(image), top_k=top_k or self.top_k)[0]
        return {
            "keypoints": result["keypoints"].detach().cpu().numpy().astype(np.float32),
            "descriptors": result["descriptors"].detach().cpu().numpy().astype(np.float32),
            "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
        }

    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        top_k: Optional[int] = None,
        min_cossim: float = 0.82,
        method: str = "sparse",
        ransac: bool = True,
        ransac_thr: float = 4.0,
        lighterglue_min_conf: float = 0.1,
    ) -> MatchResult:
        """Match two loaded OpenCV/numpy images.

        method:
            sparse: XFeat sparse keypoints + mutual nearest descriptor matching.
            semidense: XFeat* semi-dense matching.
            lighterglue: XFeat sparse keypoints + LighterGlue matcher.
        """
        top_k = top_k or self.top_k
        method = method.lower()
        if method == "sparse":
            points0, points1 = self.matcher.match_xfeat(image0, image1, top_k=top_k, min_cossim=min_cossim)
        elif method == "semidense":
            points0, points1 = self.matcher.match_xfeat_star(image0, image1, top_k=top_k)
        elif method == "lighterglue":
            desc0 = self.matcher.detectAndCompute(normalize_image(image0), top_k=top_k)[0]
            desc1 = self.matcher.detectAndCompute(normalize_image(image1), top_k=top_k)[0]
            desc0["image_size"] = (image0.shape[1], image0.shape[0])
            desc1["image_size"] = (image1.shape[1], image1.shape[0])
            points0, points1, _ = self.matcher.match_lighterglue(desc0, desc1, min_conf=lighterglue_min_conf)
        else:
            raise ValueError(f"Unsupported method '{method}'. Use sparse, semidense, or lighterglue.")

        points0 = np.asarray(points0, dtype=np.float32)
        points1 = np.asarray(points1, dtype=np.float32)
        matches = np.arange(len(points0), dtype=np.int64)[:, None].repeat(2, axis=1)
        inlier_mask = self.estimate_homography_inliers(points0, points1, ransac_thr) if ransac else None
        return MatchResult(points0=points0, points1=points1, matches=matches, inlier_mask=inlier_mask, method=method)

    def match_files(
        self,
        image0_path: str,
        image1_path: str,
        **kwargs: Any,
    ) -> MatchResult:
        image0 = read_image(image0_path)
        image1 = read_image(image1_path)
        return self.match(image0, image1, **kwargs)

    @staticmethod
    def estimate_homography_inliers(
        points0: np.ndarray,
        points1: np.ndarray,
        ransac_thr: float = 4.0,
    ) -> Optional[np.ndarray]:
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


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image


def draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    result: MatchResult,
    max_draw: int = 200,
    seed: int = 0,
) -> np.ndarray:
    indices = np.arange(result.num_matches)
    if result.inlier_mask is not None:
        indices = indices[result.inlier_mask]
    if max_draw > 0 and len(indices) > max_draw:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_draw, replace=False))

    keypoints0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in result.points0[indices]]
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in result.points1[indices]]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(indices))]
    canvas = cv2.drawMatches(
        image0,
        keypoints0,
        image1,
        keypoints1,
        matches,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    label = f"{result.method} matches={result.num_matches}"
    if result.num_inliers is not None:
        label += f" inliers={result.num_inliers}"
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the importable XFeat matching API on two image files.")
    parser.add_argument("--image0", required=True, help="Path to first image.")
    parser.add_argument("--image1", required=True, help="Path to second image.")
    parser.add_argument("--method", default="sparse", choices=["sparse", "semidense", "lighterglue"])
    parser.add_argument("--weights", default=None, help="Optional XFeat weights path. Defaults to weights/xfeat.pt.")
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--min-cossim", type=float, default=0.82)
    parser.add_argument("--no-ransac", action="store_true")
    parser.add_argument("--ransac-thr", type=float, default=4.0)
    parser.add_argument("--json", default=None, help="Optional output JSON summary path.")
    parser.add_argument("--npz", default=None, help="Optional output NPZ path with points and masks.")
    parser.add_argument("--viz", default=None, help="Optional output match visualization path.")
    parser.add_argument("--max-draw", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = XFeatMatchingAPI(weights=args.weights, top_k=args.top_k)
    image0 = read_image(args.image0)
    image1 = read_image(args.image1)
    result = api.match(
        image0,
        image1,
        top_k=args.top_k,
        min_cossim=args.min_cossim,
        method=args.method,
        ransac=not args.no_ransac,
        ransac_thr=args.ransac_thr,
    )

    summary = {
        "image0": args.image0,
        "image1": args.image1,
        "method": result.method,
        "num_matches": result.num_matches,
        "num_inliers": result.num_inliers,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.npz:
        output = Path(args.npz)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            points0=result.points0,
            points1=result.points1,
            matches=result.matches,
            inlier_mask=np.array([]) if result.inlier_mask is None else result.inlier_mask,
        )
    if args.viz:
        output = Path(args.viz)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas = draw_matches(image0, image1, result, max_draw=args.max_draw)
        if not cv2.imwrite(str(output), canvas):
            raise RuntimeError(f"Could not write visualization: {output}")


if __name__ == "__main__":
    main()
