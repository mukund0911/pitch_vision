"""
Trajectory segmentation.

Splits a continuous player track into atomic movement episodes by looking for
points where the player's speed crosses a threshold — each crossing starts a
new segment. Simple but lets us prototype the weak labeler end-to-end.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


SPEED_THRESHOLD = 1.5   # m/s — walk/jog boundary


@dataclass
class TrajectorySegment:
    player_id: int
    match_id: str
    start_frame: int
    end_frame: int
    positions: np.ndarray        # (T, 2) in field meters
    velocities: np.ndarray       # (T, 2) in m/s


class TrajectorySegmenter:
    def __init__(self, min_duration_s: float = 0.4, tracking_hz: int = 10):
        self.min_frames = max(2, int(min_duration_s * tracking_hz))
        self.hz = tracking_hz

    def segment(self, positions: np.ndarray, player_id: int,
                match_id: str) -> List[TrajectorySegment]:
        if len(positions) < self.min_frames:
            return []

        velocities = np.gradient(positions, axis=0) * self.hz
        speed = np.linalg.norm(velocities, axis=1)

        # Every time speed crosses the threshold we split the track.
        above = speed > SPEED_THRESHOLD
        breakpoints = [0]
        for i in range(1, len(above)):
            if above[i] != above[i - 1]:
                breakpoints.append(i)
        breakpoints.append(len(positions))

        segments: List[TrajectorySegment] = []
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            if end - start < self.min_frames:
                continue
            segments.append(TrajectorySegment(
                player_id=player_id,
                match_id=match_id,
                start_frame=start,
                end_frame=end,
                positions=positions[start:end].copy(),
                velocities=velocities[start:end].copy(),
            ))
        return segments
