"""Sabit hızlı, xywh durumlu Kalman filtresi."""
from __future__ import annotations

import numpy as np


# =============================================================================
# 1) KALMAN FİLTRESİ
# =============================================================================
class KalmanFilterXYWH:
  

    def __init__(self, pos_floor: float = 1.0, vel_floor: float = 0.05):
        ndim, dt = 4, 1.0
        self.F = np.eye(2 * ndim)               # durum geçiş matrisi
        for i in range(ndim):
            self.F[i, ndim + i] = dt            # x_{k+1} = x_k + v_k*dt
        self.H = np.eye(ndim, 2 * ndim)         # ölçüm matrisi: sadece [cx,cy,w,h] gözlenir
        self.std_pos = 1.0 / 20
        self.std_vel = 1.0 / 160
        self.pos_floor = pos_floor
        self.vel_floor = vel_floor

    def _std(self, w, h):
        """Oran-tabanlı gürültüye piksel tabanı uygular: max(oran*boyut, taban)."""
        sp_w = max(self.std_pos * w, self.pos_floor)
        sp_h = max(self.std_pos * h, self.pos_floor)
        sv_w = max(self.std_vel * w, self.vel_floor)
        sv_h = max(self.std_vel * h, self.vel_floor)
        return sp_w, sp_h, sv_w, sv_h

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        sp_w, sp_h, sv_w, sv_h = self._std(z[2], z[3])
        std = [2 * sp_w, 2 * sp_h, 2 * sp_w, 2 * sp_h,
               10 * sv_w, 10 * sv_h, 10 * sv_w, 10 * sv_h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        sp_w, sp_h, sv_w, sv_h = self._std(mean[2], mean[3])
        std = [sp_w, sp_h, sp_w, sp_h, sv_w, sv_h, sv_w, sv_h]
        Q = np.diag(np.square(std))             # süreç gürültüsü
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    def update(self, mean, cov, z):
        sp_w, sp_h, _, _ = self._std(mean[2], mean[3])
        R = np.diag(np.square([sp_w, sp_h, sp_w, sp_h]))  # ölçüm gürültüsü
        S = self.H @ cov @ self.H.T + R         # inovasyon kovaryansı
        K = cov @ self.H.T @ np.linalg.inv(S)   # Kalman kazancı
        y = z - self.H @ mean                   # inovasyon (ölçüm - tahmin)
        mean = mean + K @ y
        cov = (np.eye(len(mean)) - K @ self.H) @ cov
        return mean, cov
