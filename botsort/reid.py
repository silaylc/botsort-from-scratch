"""Görünüm (ReID) özniteliği."""
from __future__ import annotations

import cv2
import numpy as np


# =============================================================================
# 3) GÖRÜNÜM (ReID) ÖZNİTELİĞİ
# =============================================================================
def appearance_embedding(frame, box_xyxy):
    """Demo amaçlı basit ReID: HSV renk histogramı (L2 normalize).

    Gerçek sistemde burası bir ReID ağıdır (BoT-SORT: FastReID SBS-ResNet50).
    Arayüz aynı: kırpıntı -> sabit boyutlu, normalize vektör.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros(64, np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel()
    return (hist / (np.linalg.norm(hist) + 1e-6)).astype(np.float32)
