"""
Trajectory segmentation via changepoint detection.

Splits a continuous player track into atomic movement episodes by finding
changepoints in the speed signal. Each segment represents a roughly
consistent movement (e.g., "jogging north-east for 1.2s").

Uses PELT (ruptures library).
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TrajectorySegment:
    player_id: int
    match_id: str
    start_frame: int
    end_frame: int
    positions: np.ndarray        # (T, 2) in field meters
    velocities: np.ndarray       # (T, 2) in m/s


class TrajectorySegmenter:
    def __init__(self, min_duration_s: float = 0.4, tracking_hz: int = 10,
                 penalty: float = 3.0):
        self.min_frames = max(2, int(min_duration_s * tracking_hz))
        self.penalty = penalty
        self.hz = tracking_hz

    def segment(self, positions: np.ndarray, player_id: int,
                match_id: str) -> List[TrajectorySegment]:
        """
        Args:
            positions: (T, 2) array of (x, y) field coordinates.
        """
        if len(positions) < self.min_frames:
            return []

        import ruptures as rpt

        velocities = np.gradient(positions, axis=0) * self.hz
        speed = np.linalg.norm(velocities, axis=1).reshape(-1, 1)

        algo = rpt.Pelt(model="rbf", min_size=self.min_frames, jump=1).fit(speed)
        breakpoints = algo.predict(pen=self.penalty)

        segments: List[TrajectorySegment] = []
        prev = 0
        for bp in breakpoints:
            if bp - prev >= self.min_frames:
                segments.append(TrajectorySegment(
                    player_id=player_id,
                    match_id=match_id,
                    start_frame=prev,
                    end_frame=bp,
                    positions=positions[prev:bp].copy(),
                    velocities=velocities[prev:bp].copy(),
                ))
            prev = bp
        return segments
