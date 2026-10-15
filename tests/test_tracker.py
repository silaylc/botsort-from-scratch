import numpy as np
import pytest

from botsort import BoTSORT, KalmanFilterXYWH, iou_matrix, linear_assignment
from botsort.detectors import blob_detect


def _frame(seed=0, h=240, w=320):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, (h, w, 3), dtype=np.uint8)


def _two_targets(k):
    """İki hedef, zıt yönlerde sabit hızla hareket ediyor (xyxy)."""
    a = [20 + 3 * k, 40, 50 + 3 * k, 70]
    b = [250 - 3 * k, 150, 280 - 3 * k, 180]
    return np.array([a, b], float)


def test_iou_matrix_basic():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 3)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[0, 2] == 0


def test_linear_assignment_threshold():
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, u_r, u_c = linear_assignment(cost, thresh=0.5)
    assert matches == [(0, 0)]
    assert u_r == [1] and u_c == [1]


def test_kalman_tracks_constant_velocity():
    kf = KalmanFilterXYWH()
    mean, cov = kf.initiate(np.array([10.0, 10.0, 8.0, 8.0]))
    for k in range(1, 60):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([10.0 + 2 * k, 10.0, 8.0, 8.0]))
    # Küçük hız gürültüsü tabanı nedeniyle yakınsama yavaştır; 60 karede ~2 px/kare
    assert mean[4] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("mode", ["sort", "bytetrack", "botsort"])
def test_ids_stable_for_two_moving_targets(mode):
    tracker = BoTSORT(mode=mode)
    frame = _frame()
    ids_per_frame = []
    for k in range(30):
        tracks = tracker.update(_two_targets(k), np.array([0.9, 0.9]), frame)
        ids_per_frame.append(sorted(t.track_id for t in tracks))
    assert ids_per_frame[-1] == [1, 2]
    assert all(ids == [1, 2] for ids in ids_per_frame[1:])


def test_bytetrack_keeps_track_through_low_score_detections():
    frame = _frame()
    results = {}
    for mode in ("sort", "bytetrack"):
        tracker = BoTSORT(mode=mode)
        for k in range(10):
            tracker.update(_two_targets(k)[:1], np.array([0.9]), frame)
        # Tespit skoru düşük eşiğin altına iniyor (0.1 < 0.3 < 0.6)
        for k in range(10, 15):
            tracks = tracker.update(_two_targets(k)[:1], np.array([0.3]), frame)
        results[mode] = [t.track_id for t in tracks]
    assert results["bytetrack"] == [1]  # ikinci aşama izi korur
    assert results["sort"] == []        # SORT düşük skoru atar, iz kaybolur


def test_blob_detect_finds_bright_spot():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[40:50, 60:70] = 255
    boxes, scores = blob_detect(frame, thresh=200)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert (x1, y1) == (60, 40) and (x2, y2) == (70, 50)
    assert 0 < scores[0] < 1
