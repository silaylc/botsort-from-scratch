"""Model gerektirmeyen dedektörler: parlaklık eşikleme (blob) ve MOG2 hareket."""
import cv2
import numpy as np


# =============================================================================
# Model gerektirmeyen dedektör: parlaklık eşikleme + kontur
# =============================================================================
def blob_detect(frame, thresh=200, min_area=4, max_area=2000):
    """Termal IR'de 'sıcak' (parlak) küçük hedefleri bulur. Eğitim gerektirmez.

    1) Griye çevir (IR zaten tek kanallıdır, renkli kaydedilmişse de griye indirger)
    2) Parlaklığı `thresh` üstünde olan pikselleri beyazlat (ikili görüntü)
    3) Bu beyaz bölgelerin bağlı bileşenlerini (konturlarını) bul
    4) Alanı çok küçük (gürültü) ya da çok büyük (arka plan/false positive) olanları ele
    5) Her konturun sınırlayıcı kutusunu ve "skor"unu (ortalama parlaklık) döndür
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_brightness = cv2.mean(gray, mask=mask)[0]           # 0-255
        score = np.clip((mean_brightness - thresh) / (255 - thresh + 1e-6), 0.05, 0.99)
        boxes.append([x, y, x + w, y + h])
        scores.append(score)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    return np.array(boxes, float), np.array(scores, float)


# =============================================================================
# Model gerektirmeyen dedektör 2: zamansal arka plan çıkarma (sabit kamera)
# =============================================================================
class MotionDetector:
    """Kamera sabitken en güvenilir, eğitimsiz dedektör.

    MOG2, her piksel için bir arka plan modeli öğrenir (son `history` karedeki
    dağılımdan). Bir piksel bu modelden önemli ölçüde sapıyorsa (hareket var)
    'ön plan' (beyaz) olarak işaretlenir. Bina/duvar/pencere gibi SABİT dokular
    arka plana karışır, hiçbir zaman tespit üretmez — tek kareye bakan blob_detect'in
    aksine, bu yöntem "bu piksel hep böyleydi" bilgisini kullanır.
    """

    def __init__(self, history=60, var_thresh=16, learning_rate=0.01):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                      varThreshold=var_thresh,
                                                      detectShadows=False)
        self.lr = learning_rate
        self.n = 0

    def detect(self, frame, min_area=2, max_area=150):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fg = self.bg.apply(gray, learningRate=self.lr)
        self.n += 1
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, -1)
            strength = cv2.mean(fg, mask=mask)[0] / 255.0       # ön plan modeline uyum
            boxes.append([x, y, x + w, y + h])
            scores.append(np.clip(strength, 0.2, 0.99))
        # ilk `history` kare model ısınıyor, güvenilmez -> tespit üretme
        if self.n <= 15 or not boxes:
            return np.zeros((0, 4)), np.zeros(0)
        return np.array(boxes, float), np.array(scores, float)
