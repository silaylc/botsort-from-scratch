"""
track_video.py — bot_sort.py'yi kendi videonuzda çalıştırmak için.

ÜÇ DEDEKTÖR SEÇENEĞİ (hiçbiri eğitilmiş model gerektirmez, --detector yolo hariç):
  --detector motion  Zamansal arka plan çıkarma (MOG2). Kamera SABİTSE ve hedef küçük/
                      hareketliyse (drone sürüsü, kuş, araç) en iyi sonucu bu verir —
                      sabit duvar/pencere dokusunu yok sayar, sadece HAREKET EDENİ yakalar.
  --detector blob    Tek karede parlaklık eşikleme. Hedef arka plandan BELİRGİN ŞEKİLDE
                      daha parlaksa işe yarar (çoğu termal sahnede DEĞİL — bkz. not).
  --detector yolo     Ultralytics YOLO (eğitilmiş .pt ağırlığı). COCO ağırlığı (yolov8n.pt)
                      insan/araç gibi 80 sınıf tanır, DRONE/UAV TANIMAZ. Kendi etiketli
                      verinizle eğittiğiniz bir ağırlığınız olduğunda kullanın.

  NOT: "termal = parlak hedef" varsayımı her zaman doğru değil. Bazı IR/gece görüşü
  kayıtlarında (ör. aydınlık bir bina cephesi önünde uçan drone) hedef arka plandan
  DAHA KOYU görünür. --detector blob bu durumda işe yaramaz; --detector motion kamera
  sabitse hedefin parlak mı koyu mu olduğuna bakmaksızın çalışır çünkü sadece HAREKETİ
  yakalar.

Kurulum:
    pip install -e .                                        # motion / blob dedektörleri için yeterli
    pip install -e ".[yolo]"                                # sadece --detector yolo için

Kullanım — termal IR / küçük hedef (BUGÜN, model gerekmeden):
    python scripts/track_video.py --video "C:\\Users\\kullanici\\Downloads\\video.mp4" \
        --detector blob --thresh 200 --no-reid --match-thresh 0.6 --out cikti.mp4

Kullanım — sıradan RGB video, hazır COCO ağırlığıyla pipeline doğrulama:
    python scripts/track_video.py --video ornek.mp4 --detector yolo --weights yolov8n.pt \
        --classes 0 --out cikti.mp4

Kullanım — kendi eğittiğiniz YOLO ağırlığıyla:
    python scripts/track_video.py --video ir_video.mp4 --detector yolo --weights kendi_agirligim.pt \
        --no-reid --match-thresh 0.65 --conf 0.15 --out cikti.mp4

Notlar:
    * --no-reid: termal/gri görüntüde renk histogramı anlamsız, BoT-SORT'u Kalman+CMC
      (ReID'siz) modunda çalıştırır — resmi kodun da "BoT-SORT" (ReID'siz) varyantı budur.
    * --match-thresh: küçük/titrek hedeflerde varsayılan 0.8 çok katı, 0.5-0.65 deneyin.
    * blob dedektörü ayarları (--thresh, --min-area, --max-area) önce birkaç kareyle
      denenip videonuza göre ayarlanmalı: çıktıdaki kutular ya hiç çıkmıyorsa --thresh'i
      düşürün, ya da gürültüyü (arka plan parlaklığını) de kutuluyorsa yükseltin.
"""
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from botsort import BoTSORT
from botsort.detectors import MotionDetector, blob_detect
from botsort.visualize import draw


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="giriş video dosyası")
    ap.add_argument("--out", default="tracked_output.mp4", help="çıkış video dosyası")
    ap.add_argument("--detector", default="motion", choices=["motion", "blob", "yolo"])
    # --- motion dedektörü ayarları ---
    ap.add_argument("--history", type=int, default=60, help="[motion] arka plan hafızası (kare)")
    ap.add_argument("--var-thresh", type=float, default=16, help="[motion] hassasiyet (düşük=hassas)")
    # --- yolo dedektörü ayarları ---
    ap.add_argument("--weights", default="yolov8n.pt", help="[yolo] YOLO ağırlık dosyası")
    ap.add_argument("--conf", type=float, default=0.25, help="[yolo] dedektör güven eşiği")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="[yolo] sadece bu COCO sınıf id'lerini takip et")
    # --- blob dedektörü ayarları ---
    ap.add_argument("--thresh", type=int, default=200, help="[blob] parlaklık eşiği (0-255)")
    ap.add_argument("--min-area", type=float, default=4, help="[blob] min kontur alanı (px²)")
    ap.add_argument("--max-area", type=float, default=2000, help="[blob] max kontur alanı (px²)")
    # --- tracker ayarları ---
    ap.add_argument("--mode", default="botsort", choices=["sort", "bytetrack", "botsort"])
    ap.add_argument("--no-reid", action="store_true", help="görünüm (ReID) füzyonunu kapat")
    ap.add_argument("--no-cmc", action="store_true", help="kamera hareket telafisini kapat")
    ap.add_argument("--match-thresh", type=float, default=0.8,
                    help="1. aşama eşleme eşiği (küçük/titrek hedeflerde 0.25-0.4 deneyin)")
    ap.add_argument("--track-buffer", type=int, default=30, help="kaç kare kayıp iz tutulsun")
    ap.add_argument("--match-buffer", type=float, default=0,
                    help="SADECE eşleme hesabında kutuları bu kadar piksel büyüt (sıkışık "
                         "küçük sürülerde IoU'yu kararlılaştırır); izin GERÇEK boyutu, "
                         "görselleştirme ve kovaryans değişmez. Başlangıç için kutu "
                         "boyutunun 2-5 katı denenebilir (bkz. README 'küçük nesne' notu).")
    ap.add_argument("--kf-pos-floor", type=float, default=1.0,
                    help="Kalman'ın minimum konum gürültüsü (piksel) — küçük kutularda "
                         "oran-tabanlı gürültünün sıfıra yakın çıkmasını engeller")
    ap.add_argument("--trail-len", type=int, default=30,
                    help="her izin arkasında kaç karelik iz çizgisi (trail) gösterilsin; "
                         "0 = iz çizgisi kapalı")
    ap.add_argument("--show", action="store_true", help="işlerken bir pencerede göster")
    args = ap.parse_args()

    if args.detector == "yolo":
        from ultralytics import YOLO
        yolo_model = YOLO(args.weights)

        def detect(frame):
            r = yolo_model(frame, conf=args.conf, classes=args.classes, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            s = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            return b, s
    elif args.detector == "motion":
        motion = MotionDetector(args.history, args.var_thresh)

        def detect(frame):
            return motion.detect(frame, args.min_area, args.max_area)
    else:
        def detect(frame):
            return blob_detect(frame, args.thresh, args.min_area, args.max_area)

    tracker = BoTSORT(mode=args.mode, match_thresh=args.match_thresh,
                      track_buffer=args.track_buffer, match_buffer=args.match_buffer,
                      kf_pos_floor=args.kf_pos_floor,
                      use_reid=(args.mode == "botsort" and not args.no_reid),
                      use_cmc=(args.mode == "botsort" and not args.no_cmc))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Video açılamadı: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = defaultdict(lambda: deque(maxlen=max(args.trail_len, 1)))
    last_seen = {}

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        # --- 1) TESPİT ---
        boxes, scores = detect(frame)

        # --- 2-7) TAHMİN -> EŞLEŞTİR -> GÜNCELLE -> YÖNET (hepsi tracker.update içinde) ---
        tracks = tracker.update(boxes, scores, frame)

        if args.trail_len:
            for t in tracks:
                x1, y1, x2, y2 = t.xyxy
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[t.track_id].append((cx, cy))
                last_seen[t.track_id] = n
            # gerçekten silinmiş (uzun süre görülmemiş) izlerin kuyruğunu temizle,
            # ama kısa süreli kayıpta iz çizgisi aniden sıfırlanmasın
            for tid in list(trails):
                if n - last_seen.get(tid, n) > args.track_buffer:
                    del trails[tid]
                    del last_seen[tid]

        out_frame = draw(frame, tracks, boxes, trails if args.trail_len else None)
        writer.write(out_frame)
        if args.show:
            cv2.imshow("tracking", out_frame)
            if cv2.waitKey(1) == 27:
                break
        if n % 50 == 0:
            print(f"kare {n}  ({n / (time.time() - t0):.1f} FPS)  "
                  f"ham tespit: {len(boxes)}  aktif iz: {len(tracks)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"bitti: {n} kare -> {args.out}")


if __name__ == "__main__":
    main()
