"""İz kutuları, kimlik etiketleri ve soluklaşan iz çizgileri."""
import cv2
import numpy as np


# =============================================================================
# Görselleştirme
# =============================================================================
def id_color(i):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def draw(frame, tracks, dets=None, trails=None):
    """trails: {track_id: deque[(x,y), ...]} — her izin son birkaç karedeki merkez
    noktaları. Verilirse, her kutunun arkasına soluklaşan bir iz çizgisi çizilir."""
    img = frame.copy()
    if dets is not None:
        for b in dets:                                        # ince gri: ham tespit
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)

    if trails:
        overlay = img.copy()
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            c = id_color(tid)
            n = len(pts)
            for i in range(1, n):
                # baştan (eski) sona (yeni) doğru kalınlaşan/koyulaşan iz
                alpha = i / n
                thickness = max(1, int(round(3 * alpha)))
                cv2.line(overlay, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)      # izleri hafif saydam bindir

    for t in tracks:
        x1, y1, x2, y2 = t.xyxy.astype(int)
        c = id_color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.rectangle(img, (x1, max(y1 - 16, 0)), (x1 + 34, y1), c, -1)
        cv2.putText(img, str(t.track_id), (x1 + 3, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img
