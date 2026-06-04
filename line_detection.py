#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


def _extend_line_on_edges(edges, line):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1:
        return line

    direction = np.array([dx / length, dy / length], dtype=np.float32)
    base = np.array([float(x1), float(y1)], dtype=np.float32)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return line

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rel = pts - base
    proj = rel[:, 0] * direction[0] + rel[:, 1] * direction[1]
    perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])

    h, w = edges.shape
    dist_tol = max(2.0, min(h, w) * 0.008)
    values = np.sort(proj[perp <= dist_tol])
    if len(values) < 2:
        return line

    raw_min = min(0.0, length)
    raw_max = max(0.0, length)
    max_gap = max(5.0, min(h, w) * 0.025)

    best_cluster = None
    best_span = -1.0
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value - prev > max_gap:
            span = prev - start
            if prev >= raw_min - max_gap and start <= raw_max + max_gap and span > best_span:
                best_span = span
                best_cluster = (start, prev)
            start = value
        prev = value

    span = prev - start
    if prev >= raw_min - max_gap and start <= raw_max + max_gap and span > best_span:
        best_span = span
        best_cluster = (start, prev)

    if best_cluster is None or best_span < length * 0.75:
        return line

    start, end = best_cluster
    p1 = base + direction * start
    p2 = base + direction * end
    return (
        int(round(np.clip(p1[0], 0, w - 1))),
        int(round(np.clip(p1[1], 0, h - 1))),
        int(round(np.clip(p2[0], 0, w - 1))),
        int(round(np.clip(p2[1], 0, h - 1))),
    )


def _line_continuity_score(edges, line, band_radius=2):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1:
        return 0.0, 0.0

    h, w = edges.shape
    sample_count = max(24, int(length))
    xs = np.linspace(float(x1), float(x2), sample_count)
    ys = np.linspace(float(y1), float(y2), sample_count)

    supported = np.zeros(sample_count, dtype=bool)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        xi = int(round(x))
        yi = int(round(y))
        x_min = max(0, xi - band_radius)
        x_max = min(w, xi + band_radius + 1)
        y_min = max(0, yi - band_radius)
        y_max = min(h, yi + band_radius + 1)
        supported[idx] = np.any(edges[y_min:y_max, x_min:x_max] > 0)

    support_ratio = float(np.mean(supported))

    longest_run = 0
    current_run = 0
    for value in supported:
        if value:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    longest_run_ratio = float(longest_run / sample_count)
    return support_ratio, longest_run_ratio


def _has_large_blank_gap(edges, line, band_radius=2):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1:
        return True

    h, w = edges.shape
    sample_count = max(24, int(length))
    xs = np.linspace(float(x1), float(x2), sample_count)
    ys = np.linspace(float(y1), float(y2), sample_count)

    longest_blank = 0
    current_blank = 0
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        x_min = max(0, xi - band_radius)
        x_max = min(w, xi + band_radius + 1)
        y_min = max(0, yi - band_radius)
        y_max = min(h, yi + band_radius + 1)
        has_support = np.any(edges[y_min:y_max, x_min:x_max] > 0)
        if has_support:
            current_blank = 0
        else:
            current_blank += 1
            longest_blank = max(longest_blank, current_blank)

    max_allowed_blank = max(5, int(sample_count * 0.12))
    return longest_blank > max_allowed_blank


def _surrounding_edge_density(edges, line, inner_radius=5, outer_radius=20):
    h, w = edges.shape
    x1, y1, x2, y2 = [int(v) for v in line]

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(outer_mask, (x1, y1), (x2, y2), 255, outer_radius * 2)

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(inner_mask, (x1, y1), (x2, y2), 255, inner_radius * 2)

    surrounding_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
    surrounding_pixels = int(np.count_nonzero(surrounding_mask))
    if surrounding_pixels == 0:
        return 0.0

    edge_in_surrounding = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=surrounding_mask)))
    return float(edge_in_surrounding) / surrounding_pixels


def analyze_lines(rectified, max_candidates=3):
    """
    Analyze line candidates inside the rectified monitor image.

    Return:
      {
        "edge": edge image in rectified coordinates,
        "candidates": score-sorted candidate dictionaries,
      }
    """
    if rectified is None:
        return {"edge": None, "candidates": []}

    h, w = rectified.shape[:2]
    if h < 20 or w < 20:
        return {"edge": None, "candidates": []}

    margin_x = max(4, int(w * 0.08))
    margin_y = max(4, int(h * 0.08))
    roi = rectified[margin_y:h - margin_y, margin_x:w - margin_x]
    if roi.size == 0:
        return {"edge": None, "candidates": []}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_gray = cv2.Canny(gray, 25, 90)

    bg = np.median(roi.reshape(-1, 3), axis=0).astype(np.float32)
    dist = np.linalg.norm(roi.astype(np.float32) - bg, axis=2)
    if dist.max() > 1e-6:
        dist = np.clip(dist * (255.0 / dist.max()), 0, 255).astype(np.uint8)
    else:
        dist = np.zeros_like(gray)
    dist = cv2.GaussianBlur(dist, (3, 3), 0)
    edges_color = cv2.Canny(dist, 12, 50)

    edges = cv2.bitwise_or(edges_gray, edges_color)
    full_edges = np.zeros((h, w), dtype=np.uint8)
    full_edges[margin_y:h - margin_y, margin_x:w - margin_x] = edges

    min_side = min(h, w)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(8, int(min_side * 0.018)),
        minLineLength=max(12, int(min_side * 0.04)),
        maxLineGap=max(2, int(min_side * 0.006)),
    )
    if lines is None:
        return {"edge": full_edges, "candidates": []}

    candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        raw = (int(x1), int(y1), int(x2), int(y2))
        extended = _extend_line_on_edges(edges, raw)
        ex1, ey1, ex2, ey2 = extended
        length = math.hypot(float(ex2 - ex1), float(ey2 - ey1))

        if length < max(12, int(min_side * 0.04)):
            continue

        support_ratio, longest_run_ratio = _line_continuity_score(edges, extended)
        if support_ratio < 0.5 or longest_run_ratio < 0.62:
            continue
        if _has_large_blank_gap(edges, extended):
            continue

        surrounding_density = _surrounding_edge_density(edges, extended)
        isolation_score = max(0.0, 1.0 - surrounding_density * 3.0)

        angle = abs(calculate_angle(extended) or 0.0)
        cardinal_penalty = 0.65 if angle > 88.0 else 1.0
        score = length * cardinal_penalty * isolation_score

        shifted = (
            ex1 + margin_x,
            ey1 + margin_y,
            ex2 + margin_x,
            ey2 + margin_y,
        )
        candidates.append(
            {
                "line": shifted,
                "score": float(score),
                "length": float(length),
                "support_ratio": float(support_ratio),
                "longest_run_ratio": float(longest_run_ratio),
                "angle": calculate_angle(shifted),
                "isolation_score": float(isolation_score),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {"edge": full_edges, "candidates": candidates[:max_candidates]}


def detect_line(rectified):
    """
    Detect the longest valid line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return None if line detection fails.
    """
    analysis = analyze_lines(rectified, max_candidates=1)
    if not analysis["candidates"]:
        return None
    return analysis["candidates"][0]["line"]


def calculate_angle(line) -> Optional[float]:
    """
    Calculate the line angle in degrees.

    The convention here matches student_skeleton.py:
    vertical axis is 0 deg, left-leaning is positive, right-leaning is negative.
    """
    if line is None:
        return None

    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    angle = math.degrees(math.atan2(dx, dy))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return angle
