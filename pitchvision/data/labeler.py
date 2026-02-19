"""
Weak intent labeling from trajectory geometry.

Rather than hand-labeling data, we derive intent categories from simple
geometric rules applied to each segment:
    HOLD    — low average speed
    PRESS   — running toward the nearest opponent
    SUPPORT — running toward the ball
    SPACE   — everything else (running into open space)

These labels are noisy but cheap. They give us the training signal.
"""

from enum import IntEnum
from typing import Optional

import numpy as np


class Intent(IntEnum):
    SUPPORT = 0
    PRESS = 1
    SPACE = 2
    HOLD = 3
    UNKNOWN = 4


class WeakIntentLabeler:
    HOLD_SPEED = 0.8        # m/s
    PRESS_ANGLE = 45.0      # degrees — cone for press vs. opponent
    SUPPORT_DOT = 0.5       # cos(angle) threshold for support vs. ball

    def label(self, segment, ball_positions, opponent_positions) -> Intent:
        """
        segment: TrajectorySegment
        ball_positions: (T, 2) ball trajectory during the segment; may be empty
        opponent_positions: (M, 2) opponent positions at segment start
        """
        mean_speed = float(np.linalg.norm(segment.velocities, axis=1).mean())
        if mean_speed < self.HOLD_SPEED:
            return Intent.HOLD

        run_vec = segment.positions[-1] - segment.positions[0]
        run_norm = np.linalg.norm(run_vec)
        if run_norm < 1e-6:
            return Intent.UNKNOWN
        run_dir = run_vec / run_norm

        player_pos = segment.positions[0]

        # Press: run direction aligned with vector to nearest opponent
        if len(opponent_positions) > 0:
            dists = np.linalg.norm(opponent_positions - player_pos, axis=1)
            nearest = opponent_positions[int(dists.argmin())]
            opp_vec = nearest - player_pos
            opp_norm = np.linalg.norm(opp_vec)
            if opp_norm > 1e-6:
                opp_dir = opp_vec / opp_norm
                cos_a = float(np.clip(run_dir @ opp_dir, -1.0, 1.0))
                angle_deg = np.degrees(np.arccos(cos_a))
                if angle_deg < self.PRESS_ANGLE:
                    return Intent.PRESS

        # Support: running toward the ball
        if len(ball_positions) > 0:
            ball_vec = ball_positions[0] - player_pos
            ball_norm = np.linalg.norm(ball_vec)
            if ball_norm > 1e-6:
                ball_dir = ball_vec / ball_norm
                if float(run_dir @ ball_dir) > self.SUPPORT_DOT:
                    return Intent.SUPPORT

        return Intent.SPACE


class CausalTriggerLabeler:
    """
    Identify which opponent segment most plausibly "triggered" a player's run.

    Looks in a small window before segment start and picks the closest
    opponent who began their own segment around that time.
    """

    WINDOW_S = 0.5
    PROXIMITY_M = 15.0

    def find_trigger(self, segment, opponent_segments,
                     tracking_hz: int = 10) -> Optional[int]:
        window = int(self.WINDOW_S * tracking_hz)
        t_lo = segment.start_frame - window
        t_hi = segment.start_frame
        player_pos = segment.positions[0]

        best_id, best_dist = None, float("inf")
        for opp in opponent_segments:
            if not (t_lo <= opp.start_frame <= t_hi):
                continue
            dist = float(np.linalg.norm(opp.positions[0] - player_pos))
            if dist < self.PROXIMITY_M and dist < best_dist:
                best_dist = dist
                best_id = opp.player_id
        return best_id
