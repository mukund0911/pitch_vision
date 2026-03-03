"""
Kalman filter for bounding-box tracking.

Reference: Wojke et al. (2017), Section 2.1.

State (8D):  [cx, cy, a, h, vx, vy, va, vh]
Measurement (4D): [cx, cy, a, h]   (a = aspect ratio = w/h, h = box height)
"""

import numpy as np


class KalmanFilter:

    def __init__(self):
        ndim, dt = 4, 1.0

        # Constant-velocity motion model.
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Measurement matrix H: only observe position/size.
        self._H = np.eye(ndim, 2 * ndim)

        # Process + measurement noise, unit scale for now.
        self._Q = np.eye(2 * ndim) * 1e-2
        self._R = np.eye(ndim) * 1e-1

    def initiate(self, measurement: np.ndarray) -> tuple:
        """Create a fresh state from a new detection, with zero initial velocity."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        covariance = np.eye(8) * 10.0
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + self._Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple:
        S = self._H @ covariance @ self._H.T + self._R
        K = covariance @ self._H.T @ np.linalg.inv(S)
        y = measurement - self._H @ mean

        new_mean = mean + K @ y
        new_covariance = (np.eye(len(mean)) - K @ self._H) @ covariance
        return new_mean, new_covariance

    @staticmethod
    def bbox_to_measurement(bbox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w / max(h, 1e-6), h], dtype=np.float64)

    @staticmethod
    def measurement_to_bbox(measurement) -> list:
        cx, cy, a, h = measurement[:4]
        w = a * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [float(x1), float(y1), float(x2), float(y2)]
