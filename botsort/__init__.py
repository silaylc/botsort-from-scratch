"""
bot_sort.py — BoT-SORT'un sıfırdan, okunabilir bir uygulaması.

Makale: Aharon, Orfaig, Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian
Tracking", 2022 (arXiv:2206.14651). Resmi kod: github.com/NirAharon/BoT-SORT

Tracking-by-detection boru hattı (her karede):
    1) Tespitleri skora göre ikiye böl:  yüksek (>= track_high)  /  düşük (low..high)
    2) Tüm izlerin (aktif + kayıp) Kalman ile bir sonraki konumunu TAHMİN et
    3) [BoT-SORT] Kamera hareketini (GMC) kestir ve tahminleri dünyaya göre düzelt (CMC)
    4) 1. eşleme : izler <-> yüksek skorlu tespitler   maliyet = min(IoU-mesafe, görünüm-mesafe)
    5) 2. eşleme : kalan aktif izler <-> düşük skorlu tespitler   maliyet = IoU-mesafe  [ByteTrack fikri]
    6) Onaylanmamış (1 karelik) izler <-> kalan yüksek tespitler
    7) Eşleşmeyen yüksek tespitlerden yeni iz başlat, uzun süre kayıp izleri sil

Bu dosyada üç mod var — böylece her bileşenin katkısını ayrı ayrı görebiliriz:
    mode="sort"      : tek aşama, sadece IoU, CMC yok    (SORT benzeri)
    mode="bytetrack" : + düşük skorlu ikinci aşama       (ByteTrack)
    mode="botsort"   : + xywh Kalman, + CMC, + ReID füzyonu (BoT-SORT)
"""

from botsort.gmc import GMC, apply_cmc
from botsort.kalman import KalmanFilterXYWH
from botsort.matching import iou_matrix, linear_assignment, xywh_to_xyxy, xyxy_to_xywh
from botsort.reid import appearance_embedding
from botsort.track import Track
from botsort.tracker import BoTSORT

__all__ = [
    "BoTSORT",
    "Track",
    "KalmanFilterXYWH",
    "GMC",
    "apply_cmc",
    "appearance_embedding",
    "iou_matrix",
    "linear_assignment",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
]
