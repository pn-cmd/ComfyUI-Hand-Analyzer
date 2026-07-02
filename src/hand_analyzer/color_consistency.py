"""Color/texture consistency analysis within a hand region.

Detects "frankenstein" hands where parts of the hand have significantly
different color from the palm — a common AI-generation artifact where
fingers are pasted from different generation passes.

This module is LOCAL (no API calls) and uses OpenCV for color-space analysis.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

from .report import BoundingBox, Issue, IssueType, AnalysisSource

logger = logging.getLogger(__name__)

# Thresholds (tunable)

# Maximum allowed mean-color distance (in Lab space) between a sub-region
# and the palm's mean color. Above this, the sub-region is flagged as a
# color discontinuity.
COLOR_DISTANCE_THRESHOLD = 38.0  # Lab CIE76 distance

# Minimum sub-region area (fraction of hand mask) to check for color consistency
MIN_REGION_FRACTION = 0.05

# Number of grid subdivisions for color checking (NxN grid over the hand bbox)
_COLOR_GRID_N = 3


def analyze_color_consistency(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: BoundingBox | None = None,
) -> dict[str, Any]:
    """Analyze color consistency within a hand region.

    Divides the hand region into a grid, computes the mean Lab color of each
    grid cell (within the hand mask), and flags cells whose mean color differs
    significantly from the palm's mean color.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image (uint8).
    mask : np.ndarray
        Binary hand mask (HxW, 255=hand, 0=background).
    bbox : BoundingBox | None
        Hand bounding box (restricts analysis to this region).

    Returns
    -------
    dict with: palm_mean_color, region_colors, issues (list[Issue]).
    """
    binary = (mask > 0).astype(np.uint8)

    # Determine the analysis region
    if bbox is not None:
        x1, y1 = bbox.x, bbox.y
        x2 = min(image.shape[1], bbox.x + bbox.width)
        y2 = min(image.shape[0], bbox.y + bbox.height)
    else:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return _empty_color_result()
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # Crop to hand region
    crop_img = image[y1:y2, x1:x2]
    crop_mask = binary[y1:y2, x1:x2]

    if crop_img.size == 0 or np.sum(crop_mask > 0) < 100:
        return _empty_color_result()

    # Convert to Lab color space for perceptual color distance
    lab = cv2.cvtColor(crop_img, cv2.COLOR_RGB2LAB)

    # Compute palm mean color (center region of the mask)
    h, w = crop_mask.shape
    palm_region = crop_mask.copy()
    # Keep only the central 40% of the hand (likely palm)
    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)
    palm_region[:cy1, :] = 0
    palm_region[cy2:, :] = 0
    palm_region[:, :cx1] = 0
    palm_region[:, cx2:] = 0

    palm_pixels = lab[palm_region > 0]
    if len(palm_pixels) < 10:
        # Fallback: use entire hand
        palm_pixels = lab[crop_mask > 0]
        if len(palm_pixels) < 10:
            return _empty_color_result()

    palm_mean = np.mean(palm_pixels, axis=0)

    # Divide hand into NxN grid and check each cell
    issues: list[Issue] = []
    region_colors: list[dict[str, Any]] = []
    min_region_pixels = max(int(np.sum(crop_mask > 0) * MIN_REGION_FRACTION), 20)

    cell_h = max(h // _COLOR_GRID_N, 1)
    cell_w = max(w // _COLOR_GRID_N, 1)

    for gy in range(_COLOR_GRID_N):
        for gx in range(_COLOR_GRID_N):
            cy1 = gy * cell_h
            cy2 = min((gy + 1) * cell_h, h)
            cx1 = gx * cell_w
            cx2 = min((gx + 1) * cell_w, w)

            cell_mask = crop_mask[cy1:cy2, cx1:cx2]
            cell_pixels = lab[cy1:cy2, cx1:cx2][cell_mask > 0]

            if len(cell_pixels) < min_region_pixels:
                continue

            cell_mean = np.mean(cell_pixels, axis=0)

            # CIE76 color distance in Lab
            dist = float(np.linalg.norm(cell_mean - palm_mean))

            region_colors.append({
                "grid_pos": [gx, gy],
                "mean_color": cell_mean.tolist(),
                "distance": dist,
                "pixel_count": int(len(cell_pixels)),
            })

            if dist > COLOR_DISTANCE_THRESHOLD:
                issues.append(Issue(
                    type=IssueType.COLOR_DISCONTINUITY,
                    description=f"Sub-region at grid ({gx},{gy}) has color distance {dist:.1f} from palm (threshold {COLOR_DISTANCE_THRESHOLD}) — possible color discontinuity (frankenstein hand)",
                    confidence=min(0.8, dist / (COLOR_DISTANCE_THRESHOLD * 2)),
                    source=AnalysisSource.LANDMARK,
                    extra={
                        "grid_pos": [gx, gy],
                        "color_distance": dist,
                        "threshold": COLOR_DISTANCE_THRESHOLD,
                        "palm_mean": palm_mean.tolist(),
                        "region_mean": cell_mean.tolist(),
                    },
                ))

    return {
        "palm_mean_color": palm_mean.tolist(),
        "region_colors": region_colors,
        "issues": issues,
    }


def _empty_color_result() -> dict[str, Any]:
    return {
        "palm_mean_color": [0.0, 0.0, 0.0],
        "region_colors": [],
        "issues": [],
    }