"""
On-ball event detection from tracking data.

Purely geometric — no ML model. Works on the per-frame tracking JSON produced
by the video pipeline.

Detected events:
    PASS            — possession transfers from player A to player B, ball moves
    RECEPTION       — paired with each PASS
    SHOT            — ball moves toward goal at high speed from a possessor
    DRIBBLE         — sustained possession while the player displaces > threshold
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class EventType(Enum):
    PASS = "pass"
    SHOT = "shot"
    DRIBBLE = "dribble"
    RECEPTION = "reception"
    PRESS_TRIGGER = "press_trigger"


@dataclass
class MatchEvent:
    timestamp_s: float
    frame_idx: int
    event_type: EventType
    player_id: int
    target_player_id: Optional[int] = None
    success: Optional[bool] = None
    metadata: Optional[dict] = None


class EventDetector:
    POSSESSION_RADIUS = 3.0      # m
    BALL_SPEED_PASS = 5.0        # m/s
    BALL_SPEED_SHOT = 12.0       # m/s
    DRIBBLE_MIN_DIST = 2.0       # m
    DRIBBLE_MIN_FRAMES = 10      # = 1s at 10Hz
    GOAL_X_ABS = 52.5            # half field length
    GOAL_Y_HALF = 3.66           # half goal mouth
    SHOT_X_MIN_DIST = 30.0       # ball must be somewhat near goal's side

    def __init__(self, tracking_hz: int = 10):
        self.hz = tracking_hz

    def detect_events(self, frames: list) -> List[MatchEvent]:
        events: List[MatchEvent] = []
        possession = self._compute_possession(frames)
        events.extend(self._detect_passes(frames, possession))
        events.extend(self._detect_shots(frames, possession))
        events.extend(self._detect_dribbles(frames, possession))
        events.sort(key=lambda e: e.frame_idx)
        return events

    # -------------------------------------------------- possession per frame
    def _compute_possession(self, frames) -> list:
        possession = []
        for frame in frames:
            if frame.get("ball") is None:
                possession.append(None)
                continue
            ball_pos = np.array(frame["ball"]["pos"], dtype=np.float32)
            closest_id, closest_dist = None, float("inf")
            for player in frame.get("players", []):
                dist = float(np.linalg.norm(
                    np.array(player["pos"], dtype=np.float32) - ball_pos))
                if dist < self.POSSESSION_RADIUS and dist < closest_dist:
                    closest_dist = dist
                    closest_id = player["track_id"]
            possession.append(closest_id)
        return possession

    # -------------------------------------------------- passes
    def _detect_passes(self, frames, possession) -> List[MatchEvent]:
        events: List[MatchEvent] = []
        i = 0
        while i < len(frames) - 1:
            if possession[i] is None:
                i += 1
                continue
            passer = possession[i]
            j = i + 1
            while j < len(frames) and possession[j] is None:
                j += 1
            if j >= len(frames) or possession[j] == passer:
                i += 1
                continue

            receiver = possession[j]
            ball_i = frames[i].get("ball")
            ball_j = frames[j].get("ball")
            if ball_i is None or ball_j is None:
                i = j
                continue

            ball_dist = float(np.linalg.norm(
                np.array(ball_j["pos"]) - np.array(ball_i["pos"])))
            if ball_dist > self.POSSESSION_RADIUS:
                events.append(MatchEvent(
                    timestamp_s=frames[i]["timestamp_s"],
                    frame_idx=frames[i]["frame_idx"],
                    event_type=EventType.PASS,
                    player_id=passer,
                    target_player_id=receiver,
                    success=True,
                    metadata={"distance_m": round(ball_dist, 2)},
                ))
                events.append(MatchEvent(
                    timestamp_s=frames[j]["timestamp_s"],
                    frame_idx=frames[j]["frame_idx"],
                    event_type=EventType.RECEPTION,
                    player_id=receiver,
                ))
            i = j
        return events

    # -------------------------------------------------- shots
    def _detect_shots(self, frames, possession) -> List[MatchEvent]:
        events: List[MatchEvent] = []
        for i in range(len(frames) - 1):
            if possession[i] is None:
                continue
            b_now = frames[i].get("ball")
            b_next = frames[i + 1].get("ball")
            if b_now is None or b_next is None:
                continue

            ball_now = np.array(b_now["pos"], dtype=np.float32)
            ball_next = np.array(b_next["pos"], dtype=np.float32)
            ball_vel = (ball_next - ball_now) * self.hz
            ball_speed = float(np.linalg.norm(ball_vel))
            if ball_speed < self.BALL_SPEED_SHOT:
                continue

            # Heading toward either goal (from either half)
            heading_to_goal = (
                (ball_vel[0] > 0 and ball_now[0] > self.SHOT_X_MIN_DIST) or
                (ball_vel[0] < 0 and ball_now[0] < -self.SHOT_X_MIN_DIST)
            )
            if not heading_to_goal:
                continue

            on_target = -self.GOAL_Y_HALF <= ball_now[1] <= self.GOAL_Y_HALF
            events.append(MatchEvent(
                timestamp_s=frames[i]["timestamp_s"],
                frame_idx=frames[i]["frame_idx"],
                event_type=EventType.SHOT,
                player_id=possession[i],
                metadata={
                    "speed_ms": round(ball_speed, 2),
                    "on_target": on_target,
                },
            ))
        return events

    # -------------------------------------------------- dribbles
    def _detect_dribbles(self, frames, possession) -> List[MatchEvent]:
        events: List[MatchEvent] = []
        i = 0
        while i < len(frames):
            if possession[i] is None:
                i += 1
                continue
            player_id = possession[i]
            start = i
            while i < len(frames) and possession[i] == player_id:
                i += 1
            duration = i - start
            if duration < self.DRIBBLE_MIN_FRAMES:
                continue

            start_pos = self._player_pos(frames[start], player_id)
            end_pos = self._player_pos(frames[i - 1], player_id)
            if start_pos is None or end_pos is None:
                continue
            dist = float(np.linalg.norm(end_pos - start_pos))
            if dist >= self.DRIBBLE_MIN_DIST:
                events.append(MatchEvent(
                    timestamp_s=frames[start]["timestamp_s"],
                    frame_idx=frames[start]["frame_idx"],
                    event_type=EventType.DRIBBLE,
                    player_id=player_id,
                    metadata={
                        "distance_m": round(dist, 2),
                        "duration_s": round(duration / self.hz, 2),
                    },
                ))
        return events

    @staticmethod
    def _player_pos(frame, player_id) -> Optional[np.ndarray]:
        for p in frame.get("players", []):
            if p["track_id"] == player_id:
                return np.array(p["pos"], dtype=np.float32)
        return None
