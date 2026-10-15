"""İz (track) nesnesi."""
from __future__ import annotations

import numpy as np

from botsort.matching import xywh_to_xyxy


# =============================================================================
# 5) İZ (TRACK) NESNESİ
# =============================================================================
class Track:
    TRACKED, LOST, REMOVED = 0, 1, 2
    _next_id = 1

    def __init__(self, box_xywh, score, feat=None):
        self.box = np.asarray(box_xywh, float)
        self.score = score
        self.mean, self.cov = None, None
        self.state = Track.TRACKED
        self.is_activated = False       # 2. kez görülene kadar "onaylanmamış"
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.smooth_feat = None
        self.alpha = 0.9                # görünüm EMA katsayısı
        if feat is not None:
            self.update_features(feat)

    # --- görünüm: üstel hareketli ortalama (BoT-SORT Denk. 9) ---
    def update_features(self, feat):
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    def activate(self, kf, frame_id):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.mean, self.cov = kf.initiate(self.box)
        self.state = Track.TRACKED
        self.is_activated = frame_id == 1   # ilk karede hemen onayla
        self.frame_id = self.start_frame = frame_id
        self.tracklet_len = 0

    def update(self, kf, det, frame_id, use_feat=True):
        self.mean, self.cov = kf.update(self.mean, self.cov, det.box)
        if use_feat and det.smooth_feat is not None:
            self.update_features(det.smooth_feat)   # sadece yüksek skorlu tespitle çağrılır
        self.state = Track.TRACKED
        self.is_activated = True
        self.score = det.score
        self.frame_id = frame_id
        self.tracklet_len += 1

    def re_activate(self, kf, det, frame_id):
        self.update(kf, det, frame_id)
        self.tracklet_len = 0

    @property
    def xyxy(self):
        return xywh_to_xyxy(self.mean[:4] if self.mean is not None else self.box)
