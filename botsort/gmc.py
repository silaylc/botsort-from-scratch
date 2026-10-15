"""Kamera hareket telafisi (GMC / CMC)."""
from __future__ import annotations

import cv2
import numpy as np


# =============================================================================
# 2) KAMERA HAREKET TELAFİSİ (GMC / CMC)
# =============================================================================
class GMC:
    """Global Motion Compensation — seyrek optik akış + RANSAC ile affine kestirimi.

    Fikir: Arka plandaki köşe noktaları kamera hareketini, ön plandaki nesneler ise
    kendi hareketlerini taşır. Tespit kutularını maskeleyip sadece arka plandan
    nokta alırsak, iki kare arasındaki 2x3 affine matris A ≈ kamera hareketi olur.
    Kamera hareket ettiğinde Kalman tahmini "eski görüntü koordinatında" kalır;
    A ile çarparak tahmini yeni kareye taşırız. (Resmi kod: 'sparseOptFlow', 'orb',
    'sift', 'ecc' seçenekleri var; burada sparseOptFlow kullandık.)
    """

    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.prev_gray = None
        self.prev_pts = None

    def apply(self, frame, dets_xyxy=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.downscale > 1:
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))

        # Tespitleri maskele -> köşe noktaları sadece arka plandan alınsın
        mask = np.full_like(gray, 255)
        if dets_xyxy is not None:
            for x1, y1, x2, y2 in (np.asarray(dets_xyxy) / self.downscale).astype(int):
                mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0

        pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01,
                                      minDistance=1, blockSize=3, mask=mask)
        A = np.eye(2, 3)
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 10:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            good = status.ravel() == 1
            if good.sum() > 10:
                M, _ = cv2.estimateAffinePartial2D(self.prev_pts[good], nxt[good],
                                                   method=cv2.RANSAC)
                if M is not None:
                    A = M
                    A[:, 2] *= self.downscale   # ölçeklenmiş görüntüden orijinale
        self.prev_gray, self.prev_pts = gray, pts
        return A


def apply_cmc(tracks, A):
    """Kalman durumunu ve kovaryansını kamera affine'i ile dönüştür (BoT-SORT Denk. 6-8)."""
    R = A[:, :2]
    t = A[:, 2]
    R8 = np.kron(np.eye(4), R)                  # 8x8 blok-köşegen: her (x,y) çiftine R
    for trk in tracks:
        trk.mean = R8 @ trk.mean
        trk.mean[:2] += t                       # öteleme sadece konuma
        trk.cov = R8 @ trk.cov @ R8.T
