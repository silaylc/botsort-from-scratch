"""BoT-SORT tracker: SORT / ByteTrack / BoT-SORT modları."""
from __future__ import annotations

import numpy as np

from botsort.gmc import GMC, apply_cmc
from botsort.kalman import KalmanFilterXYWH
from botsort.matching import iou_matrix, linear_assignment, xywh_to_xyxy, xyxy_to_xywh
from botsort.reid import appearance_embedding
from botsort.track import Track


# =============================================================================
# 6) TRACKER
# =============================================================================
class BoTSORT:
    def __init__(self, mode="botsort", track_high_thresh=0.6, track_low_thresh=0.1,
                 new_track_thresh=0.7, match_thresh=0.8, track_buffer=30,
                 proximity_thresh=0.5, appearance_thresh=0.25, fuse_score=True,
                 use_cmc=None, use_reid=None, match_buffer=0.0,
                 kf_pos_floor=1.0, kf_vel_floor=0.05):
        """match_buffer: SADECE eşleme maliyetinde kutuları bu kadar piksel büyüt (her
        kenardan). İzin gerçek boyutu (Track.mean/xyxy) DEĞİŞMEZ — görselleştirme ve
        _remove_duplicates hâlâ gerçek boyutla çalışır. Çok küçük/sıkışık hedeflerde
        (ör. UAV sürüsü) IoU'nun piksel düzeyinde kırılganlığını azaltmak için var;
        yaya ölçekli sahnelerde 0 bırakılmalı (varsayılan davranışı değiştirmez)."""
        assert mode in ("sort", "bytetrack", "botsort")
        self.mode = mode
        self.use_low = mode in ("bytetrack", "botsort")    # ByteTrack katkısı
        self.use_cmc = mode == "botsort" if use_cmc is None else use_cmc     # BoT-SORT katkısı
        self.use_reid = mode == "botsort" if use_reid is None else use_reid  # BoT-SORT katkısı
        self.high, self.low = track_high_thresh, track_low_thresh
        self.new_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.buffer = track_buffer
        self.prox = proximity_thresh
        self.app = appearance_thresh
        self.fuse_score = fuse_score
        self.match_buffer = match_buffer
        self.kf = KalmanFilterXYWH(pos_floor=kf_pos_floor, vel_floor=kf_vel_floor)
        self.gmc = GMC() if self.use_cmc else None
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        Track._next_id = 1

    # --- eşleme için (gerekirse büyütülmüş) IoU; iz/tespitin gerçek boyutuna dokunmaz ---
    def _match_iou(self, t_boxes, d_boxes):
        if self.match_buffer:
            p = self.match_buffer
            pad = np.array([-p, -p, p, p])
            t_boxes = [b + pad for b in t_boxes]
            d_boxes = [b + pad for b in d_boxes]
        return iou_matrix(t_boxes, d_boxes)

    # --- maliyet matrisi: IoU (+ skor füzyonu) (+ görünüm) ---
    def _cost(self, tracks, dets, use_reid):
        ious = self._match_iou([t.xyxy for t in tracks], [xywh_to_xyxy(d.box) for d in dets])
        iou_dist = 1 - ious
        if self.fuse_score and len(dets):                   # ByteTrack'teki fuse_score
            iou_dist = 1 - ious * np.array([d.score for d in dets])[None]
        if not use_reid or iou_dist.size == 0:
            return iou_dist
        # BoT-SORT: IoU-ReID füzyonu (Denk. 12-13)
        far = (1 - ious) > self.prox                         # kutular birbirinden uzak
        T = np.stack([t.smooth_feat for t in tracks])
        D = np.stack([d.smooth_feat for d in dets])
        emb_dist = np.clip(1 - T @ D.T, 0, 2) / 2.0          # kosinüs mesafesi / 2
        emb_dist[emb_dist > self.app] = 1.0                  # görünüm yeterince benzemiyorsa reddet
        emb_dist[far] = 1.0                                  # uzaktaki kutuya görünümle atlama yasak
        return np.minimum(iou_dist, emb_dist)

    def update(self, dets_xyxy, scores, frame):
        """dets_xyxy: (N,4), scores: (N,) -> aktif izler listesi"""
        self.frame_id += 1
        dets_xyxy = np.asarray(dets_xyxy, float).reshape(-1, 4)
        scores = np.asarray(scores, float)

        activated, refind, lost, removed = [], [], [], []

        # ---- 1) Tespitleri böl ----
        hi_mask = scores >= self.high
        lo_mask = (scores > self.low) & ~hi_mask
        if not self.use_low:
            lo_mask[:] = False                                # SORT: düşük skoru tamamen at

        def make(mask, with_feat):
            out = []
            for b, s in zip(dets_xyxy[mask], scores[mask]):
                f = appearance_embedding(frame, b) if with_feat else None
                out.append(Track(xyxy_to_xywh(b), s, f))
            return out

        dets_hi = make(hi_mask, self.use_reid)
        dets_lo = make(lo_mask, False)

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]

        # ---- 2) Tahmin ----  (kayıp izler de tahmin edilir: geri bulunabilsinler)
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        for t in pool + unconfirmed:
            if t.state != Track.TRACKED:
                t.mean[7] = 0                                 # kayıpta h-hızını sıfırla
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        # ---- 3) CMC ----
        if self.use_cmc:
            A = self.gmc.apply(frame, dets_xyxy)
            apply_cmc(pool + unconfirmed, A)

        # ---- 4) 1. eşleme: yüksek skor ----
        cost = self._cost(pool, dets_hi, self.use_reid)
        matches, u_trk, u_det = linear_assignment(cost, self.match_thresh)
        for i, j in matches:
            t, d = pool[i], dets_hi[j]
            if t.state == Track.TRACKED:
                t.update(self.kf, d, self.frame_id, self.use_reid); activated.append(t)
            else:
                t.re_activate(self.kf, d, self.frame_id); refind.append(t)

        # ---- 5) 2. eşleme: düşük skor (sadece IoU, sadece aktif izler) ----
        r_tracks = [pool[i] for i in u_trk if pool[i].state == Track.TRACKED]
        cost = self._cost(r_tracks, dets_lo, False)
        if cost.size:
            cost = 1 - self._match_iou([t.xyxy for t in r_tracks],
                                       [xywh_to_xyxy(d.box) for d in dets_lo])  # skor füzyonu yok
        matches, u_trk2, _ = linear_assignment(cost, 0.5)
        for i, j in matches:
            t = r_tracks[i]
            t.update(self.kf, dets_lo[j], self.frame_id, use_feat=False)  # düşük skorla ReID güncelleme YOK
            activated.append(t)
        for i in u_trk2:
            t = r_tracks[i]
            if t.state != Track.LOST:
                t.state = Track.LOST; lost.append(t)

        # ---- 6) Onaylanmamış izler <-> kalan yüksek tespitler ----
        dets_left = [dets_hi[j] for j in u_det]
        cost = self._cost(unconfirmed, dets_left, self.use_reid)
        matches, u_unc, u_det = linear_assignment(cost, 0.7)
        for i, j in matches:
            unconfirmed[i].update(self.kf, dets_left[j], self.frame_id, self.use_reid)
            activated.append(unconfirmed[i])
        for i in u_unc:
            unconfirmed[i].state = Track.REMOVED; removed.append(unconfirmed[i])

        # ---- 7) Yeni izler ----
        for j in u_det:
            d = dets_left[j]
            if d.score >= self.new_thresh:
                d.activate(self.kf, self.frame_id); activated.append(d)

        # ---- 8) Çok uzun kayıp kalanları sil ----
        for t in self.lost:
            if self.frame_id - t.frame_id > self.buffer:
                t.state = Track.REMOVED; removed.append(t)

        # ---- Listeleri güncelle ----
        self.tracked = [t for t in self.tracked if t.state == Track.TRACKED]
        self.tracked = _union(self.tracked, activated + refind)
        self.lost = [t for t in _union(self.lost, lost) if t.state == Track.LOST
                     and t not in self.tracked]
        self.removed.extend(removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        return [t for t in self.tracked if t.is_activated]


def _union(a, b):
    seen, out = set(), []
    for t in a + b:
        if id(t) not in seen:
            seen.add(id(t)); out.append(t)
    return out


def _remove_duplicates(a, b):
    """Aktif ve kayıp listede neredeyse aynı kutu varsa, daha kısa yaşayanı at."""
    if not a or not b:
        return a, b
    d = 1 - iou_matrix([t.xyxy for t in a], [t.xyxy for t in b])
    da, db = set(), set()
    for i, j in zip(*np.where(d < 0.15)):
        ta = a[i].frame_id - a[i].start_frame
        tb = b[j].frame_id - b[j].start_frame
        (db if ta > tb else da).add(j if ta > tb else i)
    return ([t for i, t in enumerate(a) if i not in da],
            [t for j, t in enumerate(b) if j not in db])
