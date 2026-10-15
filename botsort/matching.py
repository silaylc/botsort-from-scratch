"""Kutu dönüşümleri, IoU ve Macar algoritmasıyla eşleme."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# 4) YARDIMCI FONKSİYONLAR: IoU, eşleme
# =============================================================================
def xywh_to_xyxy(b):
    cx, cy, w, h = b
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a)[:, None], np.asarray(b)[None]
    ix1, iy1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def linear_assignment(cost, thresh):
    """Macar algoritması + eşik. Dönüş: eşleşmeler, eşleşmeyen satırlar, sütunlar."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    mr = {i for i, _ in matches}
    mc = {j for _, j in matches}
    return (matches,
            [i for i in range(cost.shape[0]) if i not in mr],
            [j for j in range(cost.shape[1]) if j not in mc])
