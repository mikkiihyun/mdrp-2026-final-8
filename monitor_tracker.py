#!/usr/bin/env python3
"""
Detect and track a fixed monitor in an MP4 video, then save an annotated result.

Pipeline:
1. Initial monitor detection with edges + OpenCV LSD + quadrilateral scoring.
2. Full-scene ORB keypoint initialization.
3. LK optical-flow tracking between frames.
4. RANSAC homography estimation.
5. Predicted monitor corner transform.
6. Local Canny + Hough refinement around predicted edges.
7. Confidence scoring and re-detection on failure.

Usage:
    python monitor_tracker.py input.mp4 output_detected.mp4
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import cv2
except ImportError as exc:
    raise SystemExit(
        "필수 패키지가 설치되어 있지 않습니다.\n"
        "설치:\n"
        "    python3 -m pip install -r requirements.txt\n"
        "또는:\n"
        "    python3 -m pip install opencv-contrib-python numpy"
    ) from exc


PointArray = np.ndarray


@dataclass
class TrackResult:
    corners: PointArray | None
    confidence: float
    mode: str
    tracked_points: int = 0
    inlier_ratio: float = 0.0
    edge_score: float = 0.0
    geometry_score: float = 1.0


def dynamic_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    if median < 5:
        return cv2.Canny(gray, 50, 150)

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 50, 150
    return cv2.Canny(gray, lower, upper)


def order_corners(points: PointArray) -> PointArray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def rectify_monitor(image: np.ndarray, corners: PointArray, max_width: int = 1000) -> np.ndarray | None:
    if image is None or corners is None:
        return None

    src = order_corners(corners).astype(np.float32)
    side_lengths = [
        float(np.linalg.norm(src[(idx + 1) % 4] - src[idx]))
        for idx in range(4)
    ]

    width = max(1, int(round(max(side_lengths[0], side_lengths[2]))))
    width = min(width, max_width)
    height = max(1, int(round(width * 9.0 / 16.0)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (width, height))


def polygon_area(points: PointArray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return float(abs(cv2.contourArea(pts)))


def quad_side_lengths(points: PointArray) -> list[float]:
    pts = order_corners(points)
    return [
        float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        for i in range(4)
    ]


def quad_center(points: PointArray) -> np.ndarray:
    return np.mean(order_corners(points), axis=0)


def shrink_quad(points: PointArray, factor: float) -> PointArray:
    pts = order_corners(points)
    center = quad_center(pts)
    return (center + (pts - center) * factor).astype(np.float32)


def quad_consistency_score(candidate: PointArray, reference: PointArray, width: int, height: int) -> float:
    cand = order_corners(candidate)
    ref = order_corners(reference)
    diag = math.hypot(width, height)
    if diag <= 1e-6:
        return 0.0

    center_distance = float(np.linalg.norm(quad_center(cand) - quad_center(ref))) / diag
    position_score = 1.0 - min(1.0, center_distance / 0.20)

    cand_area = polygon_area(cand)
    ref_area = polygon_area(ref)
    if cand_area <= 1 or ref_area <= 1:
        area_score = 0.0
    else:
        area_score = min(cand_area / ref_area, ref_area / cand_area)

    cand_lengths = quad_side_lengths(cand)
    ref_lengths = quad_side_lengths(ref)
    side_ratios = [
        min(c_len / max(r_len, 1.0), r_len / max(c_len, 1.0))
        for c_len, r_len in zip(cand_lengths, ref_lengths)
    ]
    shape_score = float(np.mean(side_ratios))

    return float(max(0.0, min(1.0, 0.45 * position_score + 0.35 * area_score + 0.20 * shape_score)))


def quad_motion_score(previous: PointArray, current: PointArray, width: int, height: int) -> float:
    prev = order_corners(previous)
    curr = order_corners(current)
    diag = math.hypot(width, height)
    if diag <= 1e-6:
        return 0.0

    mean_corner_shift = float(np.mean(np.linalg.norm(curr - prev, axis=1))) / diag
    shift_score = 1.0 - min(1.0, max(0.0, mean_corner_shift - 0.015) / 0.14)

    prev_area = polygon_area(prev)
    curr_area = polygon_area(curr)
    if prev_area <= 1 or curr_area <= 1:
        area_score = 0.0
    else:
        area_score = min(prev_area / curr_area, curr_area / prev_area)

    prev_lengths = quad_side_lengths(prev)
    curr_lengths = quad_side_lengths(curr)
    side_scores = [
        min(prev_len / max(curr_len, 1.0), curr_len / max(prev_len, 1.0))
        for prev_len, curr_len in zip(prev_lengths, curr_lengths)
    ]
    side_score = float(np.mean(side_scores))

    return float(max(0.0, min(1.0, 0.45 * shift_score + 0.35 * area_score + 0.20 * side_score)))


def is_valid_quad(points: PointArray, width: int, height: int) -> bool:
    pts = order_corners(points)
    margin = 0.12 * max(width, height)
    if (
        np.any(pts[:, 0] < -margin)
        or np.any(pts[:, 0] > width + margin)
        or np.any(pts[:, 1] < -margin)
        or np.any(pts[:, 1] > height + margin)
    ):
        return False

    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False

    area = polygon_area(pts)
    image_area = float(width * height)
    if area < image_area * 0.01 or area > image_area * 0.95:
        return False

    side_lengths = [
        np.linalg.norm(pts[(i + 1) % 4] - pts[i])
        for i in range(4)
    ]
    if min(side_lengths) < min(width, height) * 0.05:
        return False

    rect_w = 0.5 * (side_lengths[0] + side_lengths[2])
    rect_h = 0.5 * (side_lengths[1] + side_lengths[3])
    if rect_w <= 1 or rect_h <= 1:
        return False

    aspect = max(rect_w / rect_h, rect_h / rect_w)
    return 1.1 <= aspect <= 5.5


def edge_density_along_quad(edges: np.ndarray, quad: PointArray, thickness: int = 5) -> float:
    mask = np.zeros(edges.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [order_corners(quad).astype(np.int32)], True, 255, thickness)
    support_pixels = int(np.count_nonzero(mask))
    if support_pixels == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)) / support_pixels)


def dark_density_along_quad(gray: np.ndarray, quad: PointArray, thickness: int = 7) -> float:
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [order_corners(quad).astype(np.int32)], True, 255, thickness)
    support_pixels = int(np.count_nonzero(mask))
    if support_pixels == 0:
        return 0.0

    dark = cv2.inRange(gray, 0, 85)
    return float(np.count_nonzero(cv2.bitwise_and(dark, dark, mask=mask)) / support_pixels)


def screen_appearance_score(gray: np.ndarray, quad: PointArray) -> float:
    pts = order_corners(quad)
    side_lengths = quad_side_lengths(pts)
    warp_w = int(max(160, min(480, 0.5 * (side_lengths[0] + side_lengths[2]))))
    warp_h = int(max(90, min(270, 0.5 * (side_lengths[1] + side_lengths[3]))))
    if warp_w <= 20 or warp_h <= 20:
        return 0.0

    dst = np.array(
        [[0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1], [0, warp_h - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, transform, (warp_w, warp_h))

    margin = max(4, int(min(warp_w, warp_h) * 0.07))
    border_mask = np.zeros((warp_h, warp_w), dtype=np.uint8)
    border_mask[:, :] = 255
    border_mask[margin:warp_h - margin, margin:warp_w - margin] = 0
    inner = warped[margin:warp_h - margin, margin:warp_w - margin]

    if inner.size == 0:
        return 0.0

    dark = cv2.inRange(warped, 0, 90)
    border_dark = np.count_nonzero(cv2.bitwise_and(dark, dark, mask=border_mask)) / max(
        1,
        np.count_nonzero(border_mask),
    )
    inner_std = float(np.std(inner))
    inner_mean = float(np.mean(inner))

    border_score = min(1.0, border_dark / 0.30)
    texture_score = min(1.0, inner_std / 28.0)
    brightness_score = 1.0 - min(1.0, abs(inner_mean - 125.0) / 125.0)

    return float(max(0.0, min(1.0, 0.55 * border_score + 0.25 * texture_score + 0.20 * brightness_score)))


def angle_score_for_quad(quad: PointArray) -> float:
    pts = order_corners(quad)
    scores = []
    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom <= 1e-6:
            return 0.0
        cosine = abs(float(np.dot(v1, v2) / denom))
        scores.append(max(0.0, 1.0 - cosine))
    return float(np.mean(scores))


def quad_score(edges: np.ndarray, gray: np.ndarray, quad: PointArray, width: int, height: int) -> float:
    pts = order_corners(quad)
    area = polygon_area(pts)
    image_area = float(width * height)
    area_score = min(1.0, area / (image_area * 0.25))
    edge_score = min(1.0, edge_density_along_quad(edges, pts, thickness=7) / 0.18)
    dark_score = min(1.0, dark_density_along_quad(gray, pts, thickness=9) / 0.22)
    screen_score = screen_appearance_score(gray, pts)
    angle_score = angle_score_for_quad(pts)

    side_lengths = [
        float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        for i in range(4)
    ]
    rect_w = 0.5 * (side_lengths[0] + side_lengths[2])
    rect_h = 0.5 * (side_lengths[1] + side_lengths[3])
    opposite_similarity = 1.0 - min(
        1.0,
        (
            abs(side_lengths[0] - side_lengths[2]) / max(side_lengths[0], side_lengths[2], 1.0)
            + abs(side_lengths[1] - side_lengths[3]) / max(side_lengths[1], side_lengths[3], 1.0)
        )
        * 0.5,
    )

    aspect = max(rect_w / rect_h, rect_h / rect_w)
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect, 1e-3) / 1.78)) / math.log(2.2))

    border_margin = 4
    border_hits = int(
        np.count_nonzero(pts[:, 0] <= border_margin)
        + np.count_nonzero(pts[:, 0] >= width - border_margin)
        + np.count_nonzero(pts[:, 1] <= border_margin)
        + np.count_nonzero(pts[:, 1] >= height - border_margin)
    )
    border_penalty = max(0.65, 1.0 - border_hits * 0.09)

    return border_penalty * (
        0.22 * edge_score
        + 0.20 * dark_score
        + 0.18 * screen_score
        + 0.15 * angle_score
        + 0.10 * aspect_score
        + 0.07 * area_score
        + 0.08 * opposite_similarity
    )


def make_lsd_mask(gray: np.ndarray, min_line_length: float) -> np.ndarray:
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except Exception:
        lsd = cv2.createLineSegmentDetector()

    detected = lsd.detect(gray)
    lines = detected[0] if detected is not None else None
    if lines is None:
        return mask

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < min_line_length:
            continue
        cv2.line(
            mask,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            255,
            2,
            cv2.LINE_AA,
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def quads_from_contours(mask: np.ndarray, min_area: float) -> list[PointArray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[PointArray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1:
            continue

        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            candidates.append(order_corners(approx.reshape(4, 2)))

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        if polygon_area(box) >= min_area:
            candidates.append(order_corners(box))

    return candidates


def y_at_x(line: np.ndarray, x: float) -> float | None:
    if abs(float(line[1])) <= 1e-6:
        return None
    return float(-(line[0] * x + line[2]) / line[1])


def x_at_y(line: np.ndarray, y: float) -> float | None:
    if abs(float(line[0])) <= 1e-6:
        return None
    return float(-(line[1] * y + line[2]) / line[0])


def cluster_similar_lines(
    line_items: list[tuple[np.ndarray, float, float]],
    center_value_index: int,
    min_separation: float,
) -> list[tuple[np.ndarray, float, float]]:
    if not line_items:
        return []

    sorted_items = sorted(line_items, key=lambda item: item[center_value_index])
    clusters: list[list[tuple[np.ndarray, float, float]]] = []

    for item in sorted_items:
        if not clusters or abs(item[center_value_index] - clusters[-1][-1][center_value_index]) > min_separation:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    merged: list[tuple[np.ndarray, float, float]] = []
    for cluster in clusters:
        total_weight = sum(item[2] for item in cluster)
        if total_weight <= 1e-6:
            continue

        line = sum((item[0] * item[2] for item in cluster), np.zeros(3, dtype=np.float32)) / total_weight
        norm = math.hypot(float(line[0]), float(line[1]))
        if norm <= 1e-6:
            continue
        line = line / norm
        center_value = sum(item[center_value_index] * item[2] for item in cluster) / total_weight
        support = sum(item[2] for item in cluster)
        merged.append((line.astype(np.float32), float(center_value), float(support)))

    return merged


def hough_rect_candidates(gray: np.ndarray, edges: np.ndarray) -> list[PointArray]:
    height, width = gray.shape[:2]
    dark = cv2.inRange(gray, 0, 90)
    dark_edges = cv2.Canny(dark, 50, 150)
    combined_edges = cv2.bitwise_or(edges, dark_edges)

    min_line = max(45, int(min(width, height) * 0.22))
    lines = cv2.HoughLinesP(
        combined_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(35, int(min(width, height) * 0.12)),
        minLineLength=min_line,
        maxLineGap=max(10, int(min(width, height) * 0.06)),
    )
    if lines is None:
        return []

    horizontal: list[tuple[np.ndarray, float, float]] = []
    vertical: list[tuple[np.ndarray, float, float]] = []
    center_x = width * 0.5
    center_y = height * 0.5

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length < min_line:
            continue

        angle = math.degrees(segment_angle(p1, p2))
        line = line_from_points(p1, p2)

        if angle <= 18 or angle >= 162:
            center_value = y_at_x(line, center_x)
            if center_value is None:
                continue
            horizontal.append((line, float(center_value), length))
        elif 72 <= angle <= 108:
            center_value = x_at_y(line, center_y)
            if center_value is None:
                continue
            vertical.append((line, float(center_value), length))

    horizontal = cluster_similar_lines(horizontal, 1, min_separation=max(8.0, height * 0.025))
    vertical = cluster_similar_lines(vertical, 1, min_separation=max(8.0, width * 0.025))

    candidates: list[PointArray] = []
    if len(horizontal) < 2 or len(vertical) < 2:
        return candidates

    horizontal = sorted(horizontal, key=lambda item: item[1])
    vertical = sorted(vertical, key=lambda item: item[1])

    for top_idx, top in enumerate(horizontal[:-1]):
        for bottom in horizontal[top_idx + 1:]:
            if bottom[1] - top[1] < height * 0.18:
                continue

            for left_idx, left in enumerate(vertical[:-1]):
                for right in vertical[left_idx + 1:]:
                    if right[1] - left[1] < width * 0.25:
                        continue

                    intersections = [
                        line_intersection(top[0], left[0]),
                        line_intersection(top[0], right[0]),
                        line_intersection(bottom[0], right[0]),
                        line_intersection(bottom[0], left[0]),
                    ]
                    if any(point is None for point in intersections):
                        continue

                    quad = order_corners(np.array(intersections, dtype=np.float32))
                    if not is_valid_quad(quad, width, height):
                        continue

                    candidates.append(quad)

    return candidates


def line_from_points(p1: PointArray, p2: PointArray) -> np.ndarray:
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def line_distance(line: np.ndarray, points: PointArray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    values = line[0] * pts[:, 0] + line[1] * pts[:, 1] + line[2]
    return float(np.mean(np.abs(values)))


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    cross = np.cross(line1, line2)
    if abs(float(cross[2])) <= 1e-6:
        return None
    return np.array([cross[0] / cross[2], cross[1] / cross[2]], dtype=np.float32)


def segment_angle(p1: PointArray, p2: PointArray) -> float:
    angle = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
    if angle < 0:
        angle += math.pi
    if angle >= math.pi:
        angle -= math.pi
    return angle


def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, math.pi - diff)


class MonitorTracker:
    def __init__(
        self,
        max_features: int = 3000,
        detector_width: int = 960,
        local_band: int = 20,
        redetect_confidence: float = 0.5,
        min_track_points: int = 80,
    ) -> None:
        self.max_features = max_features
        self.detector_width = detector_width
        self.local_band = local_band
        self.redetect_confidence = redetect_confidence
        self.min_track_points = min_track_points

        self.orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: PointArray | None = None
        self.corners: PointArray | None = None
        self.last_good_corners: PointArray | None = None
        self.corner_history: deque[PointArray] = deque(maxlen=10)
        self.frame_index = 0

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.corners = None
        self.corner_history.clear()

    def initial_detect(self, frame: np.ndarray, expected: PointArray | None = None) -> PointArray | None:
        height, width = frame.shape[:2]
        scale = 1.0
        work = frame
        expected_work = None

        if self.detector_width > 0 and width > self.detector_width:
            scale = self.detector_width / float(width)
            work = cv2.resize(frame, (self.detector_width, int(round(height * scale))))
            if expected is not None:
                expected_work = order_corners(expected * scale)
        elif expected is not None:
            expected_work = order_corners(expected)

        work_h, work_w = work.shape[:2]
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = dynamic_canny(gray)

        min_area = float(work_w * work_h) * 0.015
        min_line_length = float(min(work_w, work_h)) * 0.08

        lsd_mask = make_lsd_mask(gray, min_line_length=min_line_length)
        edge_mask = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        combined = cv2.bitwise_or(edge_mask, lsd_mask)
        candidates = []
        candidates.extend(hough_rect_candidates(gray, edges))
        candidates.extend(quads_from_contours(lsd_mask, min_area))
        candidates.extend(quads_from_contours(edge_mask, min_area))
        candidates.extend(quads_from_contours(combined, min_area))

        best_quad = None
        best_score = 0.0
        seen: set[tuple[int, ...]] = set()

        for candidate in candidates:
            candidate = order_corners(candidate)
            key = tuple(np.round(candidate.reshape(-1) / 4.0).astype(int).tolist())
            if key in seen:
                continue
            seen.add(key)

            if not is_valid_quad(candidate, work_w, work_h):
                continue

            score = quad_score(edges, gray, candidate, work_w, work_h)
            if expected_work is not None:
                temporal_score = quad_consistency_score(candidate, expected_work, work_w, work_h)
                score = 0.76 * score + 0.24 * temporal_score
            if score > best_score:
                best_score = score
                best_quad = candidate

        if best_quad is None or best_score < 0.32:
            return None

        return order_corners(best_quad / scale)

    def feature_mask(self, gray: np.ndarray, monitor_quad: PointArray | None = None) -> np.ndarray | None:
        if monitor_quad is None:
            return None

        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
        inner = shrink_quad(monitor_quad, 0.88)
        cv2.fillConvexPoly(mask, inner.astype(np.int32), 0)
        return mask

    def seed_features(self, gray: np.ndarray, monitor_quad: PointArray | None = None) -> PointArray | None:
        mask = self.feature_mask(gray, monitor_quad)
        keypoints = self.orb.detect(gray, mask)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

        points = np.array([kp.pt for kp in keypoints[: self.max_features]], dtype=np.float32)
        if len(points) < self.min_track_points:
            fallback = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_features,
                qualityLevel=0.01,
                minDistance=7,
                mask=mask,
                blockSize=7,
            )
            if fallback is None:
                return None
            points = fallback.reshape(-1, 2).astype(np.float32)

        if len(points) == 0:
            return None
        return points.reshape(-1, 1, 2)

    def initialize(self, frame: np.ndarray) -> TrackResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = self.initial_detect(frame, self.last_good_corners)
        if corners is None:
            self.reset()
            self.prev_gray = gray
            return TrackResult(None, 0.0, "NO_DETECTION")

        self.corners = corners
        self.last_good_corners = corners.copy()
        self.prev_gray = gray
        self.prev_pts = self.seed_features(gray, corners)
        self.corner_history.clear()
        self.corner_history.append(corners.copy())

        tracked_points = 0 if self.prev_pts is None else len(self.prev_pts)
        return TrackResult(corners, 1.0, "INITIAL_DETECT", tracked_points, 1.0, 1.0)

    def local_refine(self, gray: np.ndarray, predicted: PointArray) -> tuple[PointArray, float]:
        height, width = gray.shape[:2]
        predicted = order_corners(predicted)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = dynamic_canny(blurred)

        refined_lines: list[np.ndarray | None] = []
        side_edge_scores: list[float] = []

        for i in range(4):
            p1 = predicted[i]
            p2 = predicted[(i + 1) % 4]
            pred_angle = segment_angle(p1, p2)
            pred_len = float(np.linalg.norm(p2 - p1))
            if pred_len < 10:
                refined_lines.append(None)
                side_edge_scores.append(0.0)
                continue

            x_min = int(max(0, math.floor(min(p1[0], p2[0]) - self.local_band)))
            y_min = int(max(0, math.floor(min(p1[1], p2[1]) - self.local_band)))
            x_max = int(min(width, math.ceil(max(p1[0], p2[0]) + self.local_band)))
            y_max = int(min(height, math.ceil(max(p1[1], p2[1]) + self.local_band)))

            if x_max <= x_min + 2 or y_max <= y_min + 2:
                refined_lines.append(None)
                side_edge_scores.append(0.0)
                continue

            roi_edges = edges[y_min:y_max, x_min:x_max]
            lines = cv2.HoughLinesP(
                roi_edges,
                rho=1,
                theta=np.pi / 180.0,
                threshold=max(12, int(pred_len * 0.08)),
                minLineLength=max(18, int(pred_len * 0.35)),
                maxLineGap=max(8, self.local_band),
            )

            predicted_line = line_from_points(p1, p2)
            best_line = None
            best_score = -1.0

            if lines is not None:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    gp1 = np.array([x1 + x_min, y1 + y_min], dtype=np.float32)
                    gp2 = np.array([x2 + x_min, y2 + y_min], dtype=np.float32)
                    seg_len = float(np.linalg.norm(gp2 - gp1))
                    if seg_len < pred_len * 0.25:
                        continue

                    diff = angle_diff(pred_angle, segment_angle(gp1, gp2))
                    if diff > math.radians(14):
                        continue

                    candidate_line = line_from_points(gp1, gp2)
                    dist = line_distance(candidate_line, np.array([p1, p2], dtype=np.float32))
                    if dist > self.local_band * 1.5:
                        continue

                    angle_score = 1.0 - diff / math.radians(14)
                    distance_score = 1.0 - min(1.0, dist / max(1.0, self.local_band * 1.5))
                    length_score = min(1.0, seg_len / pred_len)
                    score = 0.45 * angle_score + 0.35 * distance_score + 0.20 * length_score

                    if score > best_score:
                        best_score = score
                        best_line = candidate_line

            refined_lines.append(best_line)
            if best_line is None:
                side_edge_scores.append(0.0)
            else:
                side_edge_scores.append(max(0.0, min(1.0, best_score)))

        if sum(line is not None for line in refined_lines) < 3:
            edge_score = edge_density_along_quad(edges, predicted, thickness=7)
            return predicted, min(1.0, edge_score / 0.18)

        final_lines = []
        for i, line in enumerate(refined_lines):
            if line is None:
                final_lines.append(line_from_points(predicted[i], predicted[(i + 1) % 4]))
            else:
                final_lines.append(line)

        intersections = [
            line_intersection(final_lines[3], final_lines[0]),
            line_intersection(final_lines[0], final_lines[1]),
            line_intersection(final_lines[1], final_lines[2]),
            line_intersection(final_lines[2], final_lines[3]),
        ]

        if any(point is None for point in intersections):
            edge_score = edge_density_along_quad(edges, predicted, thickness=7)
            return predicted, min(1.0, edge_score / 0.18)

        refined = order_corners(np.array(intersections, dtype=np.float32))
        if not is_valid_quad(refined, width, height):
            edge_score = edge_density_along_quad(edges, predicted, thickness=7)
            return predicted, min(1.0, edge_score / 0.18)

        displacement = float(np.mean(np.linalg.norm(refined - predicted, axis=1)))
        if displacement > self.local_band * 2.5:
            edge_score = edge_density_along_quad(edges, predicted, thickness=7)
            return predicted, min(1.0, edge_score / 0.18)

        refined = 0.70 * refined + 0.30 * predicted
        return refined.astype(np.float32), float(np.mean(side_edge_scores))

    def stability_score(self) -> float:
        if len(self.corner_history) < 3:
            return 1.0

        stack = np.stack(list(self.corner_history), axis=0)
        per_corner_std = np.linalg.norm(np.std(stack, axis=0), axis=1)
        mean_std = float(np.mean(per_corner_std))
        return max(0.0, min(1.0, 1.0 - mean_std / 18.0))

    def confidence(
        self,
        tracked_points: int,
        previous_points: int,
        inlier_ratio: float,
        edge_score: float,
        geometry_score: float,
    ) -> float:
        feature_score = min(1.0, tracked_points / max(float(previous_points), self.min_track_points))
        score = (
            0.20 * feature_score
            + 0.26 * inlier_ratio
            + 0.34 * edge_score
            + 0.12 * geometry_score
            + 0.08 * self.stability_score()
        )
        if edge_score < 0.12:
            score = min(score, 0.44)
        if geometry_score < 0.35:
            score = min(score, 0.48)
        return float(max(0.0, min(1.0, score)))

    def smooth_corners(self, candidate: PointArray, confidence: float, geometry_score: float) -> PointArray:
        if self.corners is None:
            return order_corners(candidate)

        alpha = 0.32 + 0.48 * confidence * geometry_score
        alpha = float(max(0.25, min(0.82, alpha)))
        return (alpha * order_corners(candidate) + (1.0 - alpha) * order_corners(self.corners)).astype(np.float32)

    def track(self, frame: np.ndarray) -> TrackResult:
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.corners is None or self.prev_pts is None:
            return self.initialize(frame)

        previous_count = len(self.prev_pts)
        if previous_count < self.min_track_points:
            self.prev_pts = self.seed_features(self.prev_gray, self.corners)
            previous_count = 0 if self.prev_pts is None else len(self.prev_pts)
            if self.prev_pts is None or previous_count < self.min_track_points:
                return self.initialize(frame)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            **self.lk_params,
        )

        if curr_pts is None or status is None:
            return self.initialize(frame)

        status = status.reshape(-1).astype(bool)
        good_prev = self.prev_pts.reshape(-1, 2)[status]
        good_curr = curr_pts.reshape(-1, 2)[status]

        if len(good_curr) < self.min_track_points:
            return self.initialize(frame)

        homography, inlier_mask = cv2.findHomography(
            good_prev,
            good_curr,
            cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=2000,
            confidence=0.995,
        )

        if homography is None or inlier_mask is None:
            return self.initialize(frame)

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / max(1, len(good_curr))

        if inlier_count < self.min_track_points * 0.5 or inlier_ratio < 0.25:
            return self.initialize(frame)

        predicted = cv2.perspectiveTransform(
            self.corners.reshape(1, 4, 2).astype(np.float32),
            homography,
        ).reshape(4, 2)

        frame_h, frame_w = frame.shape[:2]
        if not is_valid_quad(predicted, frame_w, frame_h):
            return self.initialize(frame)

        refined, edge_score = self.local_refine(gray, predicted)
        geometry_score = quad_motion_score(self.corners, refined, frame_w, frame_h)
        conf = self.confidence(inlier_count, previous_count, inlier_ratio, edge_score, geometry_score)

        if conf < self.redetect_confidence or edge_score < 0.12 or geometry_score < 0.35:
            redetected = self.initial_detect(frame, refined)
            if redetected is not None:
                self.corners = redetected
                self.last_good_corners = redetected.copy()
                self.prev_gray = gray
                self.prev_pts = self.seed_features(gray, redetected)
                self.corner_history.clear()
                self.corner_history.append(redetected.copy())
                count = 0 if self.prev_pts is None else len(self.prev_pts)
                return TrackResult(redetected, max(conf, 0.65), "REDETECT", count, inlier_ratio, edge_score, 1.0)

            if self.corners is not None and (conf < self.redetect_confidence or geometry_score < 0.25):
                held = self.corners.copy()
                self.prev_gray = gray
                self.prev_pts = self.seed_features(gray, held)
                count = 0 if self.prev_pts is None else len(self.prev_pts)
                return TrackResult(held, min(conf, 0.35), "HOLD_LAST", count, inlier_ratio, edge_score, geometry_score)

        smoothed = self.smooth_corners(refined, conf, geometry_score)
        self.corners = smoothed
        if conf >= self.redetect_confidence and edge_score >= 0.12 and geometry_score >= 0.35:
            self.last_good_corners = smoothed.copy()
        self.prev_gray = gray
        self.prev_pts = good_curr[inliers].reshape(-1, 1, 2).astype(np.float32)

        if len(self.prev_pts) < self.max_features * 0.35 or self.frame_index % 45 == 0:
            reseeded = self.seed_features(gray, smoothed)
            if reseeded is not None and len(reseeded) >= len(self.prev_pts):
                self.prev_pts = reseeded

        self.corner_history.append(smoothed.copy())
        mode = "TRACK" if conf >= self.redetect_confidence else "LOW_CONFIDENCE"
        return TrackResult(smoothed, conf, mode, inlier_count, inlier_ratio, edge_score, geometry_score)


def draw_result(
    frame: np.ndarray,
    result: TrackResult,
    draw_features: bool = False,
    feature_points: PointArray | None = None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    if result.corners is not None:
        if result.confidence >= 0.7:
            color = (40, 220, 40)
        elif result.confidence >= 0.5:
            color = (0, 210, 255)
        else:
            color = (0, 80, 255)

        corners = order_corners(result.corners).astype(np.int32)
        cv2.polylines(output, [corners], True, color, 3, cv2.LINE_AA)

        labels = ["P1", "P2", "P3", "P4"]
        for idx, point in enumerate(corners):
            cv2.circle(output, tuple(point), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, tuple(point), 5, color, 2, cv2.LINE_AA)
            cv2.putText(
                output,
                labels[idx],
                (int(point[0]) + 7, int(point[1]) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    if draw_features and feature_points is not None:
        for point in feature_points.reshape(-1, 2)[:500]:
            cv2.circle(output, tuple(np.round(point).astype(int)), 1, (255, 180, 60), -1)

    info_primary = f"{result.mode}  conf={result.confidence:.2f}"
    info_secondary = (
        f"pts={result.tracked_points}  inliers={result.inlier_ratio:.2f}  "
        f"edge={result.edge_score:.2f}  geom={result.geometry_score:.2f}"
    )
    cv2.rectangle(output, (10, 10), (min(width - 10, 620), 66), (0, 0, 0), -1)
    cv2.putText(
        output,
        info_primary,
        (22, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        info_secondary,
        (22, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result.corners is None:
        cv2.putText(
            output,
            "monitor not found",
            (22, height - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def make_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {output_path}")
    return writer


def verification_frame_indices(total_frames: int) -> set[int]:
    if total_frames > 0:
        return {0, total_frames // 4, total_frames // 2, (total_frames * 3) // 4, total_frames - 1}
    return {0, 150, 300, 450, 600}


def write_preview_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        small = cv2.resize(image, (320, 180))
        images.append(small)

    if not images:
        return

    sheet = np.concatenate(images, axis=1)
    cv2.imwrite(str(output_path), sheet)


def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_detected.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verification_dir = Path(args.verification_dir) if args.verification_dir else None
    verification_targets: set[int] = set()
    verification_images: list[Path] = []
    if verification_dir is not None:
        verification_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3 or math.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    verification_targets = verification_frame_indices(total_frames)

    writer = make_video_writer(output_path, fps, (width, height))
    tracker = MonitorTracker(
        max_features=args.max_features,
        detector_width=args.detector_width,
        local_band=args.local_band,
        redetect_confidence=args.redetect_confidence,
        min_track_points=args.min_track_points,
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = tracker.track(frame)
            annotated = draw_result(
                frame,
                result,
                draw_features=args.draw_features,
                feature_points=tracker.prev_pts,
            )
            writer.write(annotated)

            if verification_dir is not None and frame_idx in verification_targets:
                image_path = verification_dir / f"{output_path.stem}_frame_{frame_idx:06d}.jpg"
                verification_image = annotated.copy()
                cv2.putText(
                    verification_image,
                    f"frame {frame_idx}",
                    (14, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(image_path), verification_image)
                verification_images.append(image_path)

            if args.show:
                cv2.imshow("monitor tracker", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                if total_frames > 0:
                    print(f"processed {frame_idx}/{total_frames} frames")
                else:
                    print(f"processed {frame_idx} frames")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if verification_dir is not None:
        preview_path = verification_dir / f"{output_path.stem}_preview.jpg"
        write_preview_sheet(verification_images, preview_path)
        print(f"verification images: {verification_dir}")

    print(f"saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track monitor corners in an MP4 and save an annotated output video.",
    )
    parser.add_argument("input", help="Input MP4/video path")
    parser.add_argument("output", nargs="?", help="Output MP4 path. Default: <input>_detected.mp4")
    parser.add_argument("--max-features", type=int, default=3000, help="ORB features for LK tracking")
    parser.add_argument("--detector-width", type=int, default=960, help="Resize width for initial detection")
    parser.add_argument("--local-band", type=int, default=20, help="Pixel band around each predicted edge")
    parser.add_argument("--redetect-confidence", type=float, default=0.50, help="Re-detect below this confidence")
    parser.add_argument("--min-track-points", type=int, default=80, help="Minimum tracked inlier points")
    parser.add_argument("--draw-features", action="store_true", help="Draw tracked scene features")
    parser.add_argument("--verification-dir", help="Directory for sampled verification JPGs and preview sheet")
    parser.add_argument("--show", action="store_true", help="Preview while processing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
