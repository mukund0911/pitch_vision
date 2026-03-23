"""
MatchAccumulator — per-player statistics rolled up across a whole match.

Combines three signal sources:
    1. On-ball events from EventDetector.
    2. Off-ball intent predictions from PitchVisionModel.
    3. Per-frame kinematics computed from tracking positions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PlayerProfile:
    player_id: int = -1
    # On-ball
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    shots_on_target: int = 0
    dribbles: int = 0
    dribble_distance_m: float = 0.0
    receptions: int = 0
    # Off-ball
    support_runs: int = 0
    pressing_actions: int = 0
    space_creating_runs: int = 0
    hold_actions: int = 0
    avg_urgency: float = 0.0
    # Kinematics
    distance_km: float = 0.0
    sprints: int = 0
    top_speed_ms: float = 0.0
    avg_speed_ms: float = 0.0

    @property
    def pass_completion(self) -> float:
        return self.passes_completed / max(self.passes_attempted, 1)


class MatchAccumulator:
    SPRINT_SPEED = 7.0  # m/s

    def __init__(self):
        self.profiles: Dict[int, PlayerProfile] = defaultdict(PlayerProfile)
        self.events: List[dict] = []
        self._urgency: Dict[int, list] = defaultdict(list)
        self._speeds: Dict[int, list] = defaultdict(list)

    # ------------------------------------------------- on-ball
    def add_events(self, events):
        for e in events:
            pid = e.player_id
            self.profiles[pid].player_id = pid
            self.events.append({
                "timestamp_s": e.timestamp_s,
                "type": e.event_type.value,
                "player_id": pid,
                "target_player_id": e.target_player_id,
                "success": e.success,
                "metadata": e.metadata,
            })

            t = e.event_type.value
            if t == "pass":
                self.profiles[pid].passes_attempted += 1
                if e.success:
                    self.profiles[pid].passes_completed += 1
            elif t == "shot":
                self.profiles[pid].shots += 1
                if e.metadata and e.metadata.get("on_target"):
                    self.profiles[pid].shots_on_target += 1
            elif t == "dribble":
                self.profiles[pid].dribbles += 1
                if e.metadata:
                    self.profiles[pid].dribble_distance_m += e.metadata.get("distance_m", 0)
            elif t == "reception":
                self.profiles[pid].receptions += 1

    # ------------------------------------------------- off-ball
    def add_intent(self, player_id: int, intent: str, urgency: float):
        p = self.profiles[player_id]
        p.player_id = player_id
        if intent == "support":
            p.support_runs += 1
        elif intent == "press":
            p.pressing_actions += 1
        elif intent == "space":
            p.space_creating_runs += 1
        elif intent == "hold":
            p.hold_actions += 1
        self._urgency[player_id].append(urgency)

    # ------------------------------------------------- kinematics
    def add_kinematics(self, player_id: int, speed: float, dt: float):
        p = self.profiles[player_id]
        p.player_id = player_id
        p.distance_km += speed * dt / 1000.0
        self._speeds[player_id].append(speed)
        if speed > p.top_speed_ms:
            p.top_speed_ms = speed
        if speed > self.SPRINT_SPEED:
            p.sprints += 1

    def finalize(self) -> Dict[int, PlayerProfile]:
        for pid, vals in self._urgency.items():
            if vals:
                self.profiles[pid].avg_urgency = float(np.mean(vals))
        for pid, vals in self._speeds.items():
            if vals:
                self.profiles[pid].avg_speed_ms = float(np.mean(vals))
        return dict(self.profiles)

    def event_log(self) -> list:
        return sorted(self.events, key=lambda e: e["timestamp_s"])

    def player(self, player_id: int) -> dict:
        profile = self.profiles.get(player_id)
        if profile is None:
            return {"error": f"player {player_id} not found"}
        events = [e for e in self.events if e["player_id"] == player_id]
        return {"profile": vars(profile), "events": events}
