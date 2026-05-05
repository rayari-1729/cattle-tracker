"""Lightweight cattle re-identification via HSV color histograms."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

H_BINS = 12
S_BINS = 6
V_BINS = 6


def fingerprint(crop_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV color histogram (length H_BINS*S_BINS*V_BINS)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(H_BINS * S_BINS * V_BINS, dtype=np.float32)
    img = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32).flatten()
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection similarity in [0, 1]."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.minimum(a, b).sum())


def best_match(
    query: np.ndarray, herd: list[dict], threshold: float = 0.55
) -> Optional[dict]:
    """Find the best matching cow in the herd. Returns {cow, score} or None."""
    best = None
    best_score = 0.0
    for cow in herd:
        prints = cow.get("fingerprints", [])
        if not prints:
            continue
        for fp in prints:
            arr = np.asarray(fp, dtype=np.float32)
            s = similarity(query, arr)
            if s > best_score:
                best_score = s
                best = cow
    if best is not None and best_score >= threshold:
        return {"cow": best, "score": best_score}
    return None
