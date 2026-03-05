"""
Track — one tracked object with its Kalman state and appearance gallery.

Lifecycle:
    TENTATIVE -> CONFIRMED (after n_init consecutive hits)
    CONFIRMED -> DELETED   (after time_since_update > max_age)
    TENTATIVE -> DELETED   (on the first missed frame; likely a false positive)
"""

import numpy as np

from .kalman import KalmanFilter


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    _next_id = 1
    FEATURE_GALLERY_SIZE = 100

    def __init__(self, mean, covariance, n_init: int = 3,
                 max_age: int = 30, feature=None):
        self.track_id = Track._next_id
        Track._next_id += 1

        self.mean = mean
        self.covariance = covariance

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.n_init = n_init
        self.max_age = max_age

        self.features = []
        if feature is not None:
            self.features.append(np.asarray(feature, dtype=np.float32))

    def predict(self, kalman_filter: KalmanFilter):
        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kalman_filter: KalmanFilter, detection_bbox, feature=None):
        measurement = KalmanFilter.bbox_to_measurement(detection_bbox)
        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0

        if feature is not None:
            self.features.append(np.asarray(feature, dtype=np.float32))
            if len(self.features) > self.FEATURE_GALLERY_SIZE:
                self.features.pop(0)

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def to_bbox(self) -> list:
        return KalmanFilter.measurement_to_bbox(self.mean[:4])
