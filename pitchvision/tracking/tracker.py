"""
DeepSORT tracker — the full assembly.

Reference: Wojke et al. (2017).

Per-frame update:
    1. Kalman predict on every existing track.
    2. Extract ReID embeddings for each detection (one crop each).
    3. Appearance matching — confirmed tracks vs detections, cosine-gated.
    4. IoU matching — remaining tracks (including tentative) vs remaining detections.
    5. Update matched tracks (Kalman update + append feature).
    6. Mark unmatched tracks as missed; create new tracks from unmatched detections.
    7. Delete tracks whose state says so.
"""

import numpy as np

from .kalman import KalmanFilter
from .reid import ReIDNet
from .track import Track, TrackState
from .matching import (
    cosine_distance,
    iou_distance,
    gate_cost_matrix,
    hungarian_match,
)


class DeepSORTTracker:

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_cosine_distance: float = 0.3,
                 max_iou_distance: float = 0.7,
                 reid_model: ReIDNet = None):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance

        self.kalman_filter = KalmanFilter()
        self.reid = reid_model if reid_model is not None else ReIDNet()
        self.reid.eval()
        self.tracks = []

    # ------------------------------------------------------------------ main
    def update(self, detections: list, frame: np.ndarray) -> list:
        # 1) Predict
        for track in self.tracks:
            track.predict(self.kalman_filter)

        # 2) Extract appearance features for each detection
        det_bboxes = [d[:4] for d in detections]
        crops = [self._crop(frame, bb) for bb in det_bboxes]
        features = self.reid.extract(crops)  # (M, 128)

        # 3) Appearance matching on confirmed tracks
        conf_idx = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconf_idx = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_conf, unmatched_dets = self._appearance_match(
            conf_idx, features)

        # 4) IoU matching on (tentative tracks + confirmed tracks that missed
        #    appearance matching) vs remaining detections
        iou_track_idx = unconf_idx + unmatched_conf
        matches_b, unmatched_iou, unmatched_dets = self._iou_match(
            iou_track_idx, unmatched_dets, det_bboxes)

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_iou  # what remains after both stages

        # 5) Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                self.kalman_filter,
                det_bboxes[det_idx],
                feature=features[det_idx] if len(features) else None,
            )

        # 6) Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 7) Spawn new tracks from unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(
                det_bboxes[det_idx],
                feature=features[det_idx] if len(features) else None,
            )

        # 8) Drop deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return [
            {"track_id": int(t.track_id), "bbox": t.to_bbox()}
            for t in self.tracks if t.is_confirmed()
        ]

    # ------------------------------------------------------------------ matching helpers
    def _appearance_match(self, track_idx: list, det_features: np.ndarray):
        """Match confirmed tracks to detections by cosine distance of ReID features."""
        if not track_idx or det_features.shape[0] == 0:
            return [], list(track_idx), list(range(det_features.shape[0]))

        track_features = np.stack(
            [self._mean_feature(self.tracks[i]) for i in track_idx], axis=0)
        cost = cosine_distance(track_features, det_features)
        cost = gate_cost_matrix(cost, self.max_cosine_distance)

        matches, unmatched_rows, unmatched_cols = hungarian_match(cost)

        matches = [(track_idx[r], c) for r, c in matches]
        unmatched_tracks = [track_idx[r] for r in unmatched_rows]
        return matches, unmatched_tracks, unmatched_cols

    def _iou_match(self, track_idx: list, det_idx: list, det_bboxes: list):
        """Match by IoU on predicted-vs-detection box overlap."""
        if not track_idx or not det_idx:
            return [], list(track_idx), list(det_idx)

        track_bboxes = [self.tracks[i].to_bbox() for i in track_idx]
        sub_det_bboxes = [det_bboxes[j] for j in det_idx]

        cost = iou_distance(track_bboxes, sub_det_bboxes)
        cost = gate_cost_matrix(cost, self.max_iou_distance)

        matches, unmatched_rows, unmatched_cols = hungarian_match(cost)

        matches = [(track_idx[r], det_idx[c]) for r, c in matches]
        unmatched_tracks = [track_idx[r] for r in unmatched_rows]
        unmatched_dets = [det_idx[c] for c in unmatched_cols]
        return matches, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _mean_feature(track: Track) -> np.ndarray:
        """Average (and re-normalize) the feature gallery for a track."""
        feats = np.stack(track.features, axis=0)
        mean = feats.mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-6
        return mean / norm

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((128, 64, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _initiate_track(self, bbox, feature):
        measurement = KalmanFilter.bbox_to_measurement(bbox)
        mean, covariance = self.kalman_filter.initiate(measurement)
        self.tracks.append(Track(
            mean, covariance,
            n_init=self.n_init, max_age=self.max_age,
            feature=feature,
        ))
