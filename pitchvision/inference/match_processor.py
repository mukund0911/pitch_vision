"""
MatchProcessor — end-to-end video → scouting report.

Steps:
    1. VideoProcessor: detect + track + homography → per-frame tracking JSON.
    2. EventDetector: heuristic on-ball events.
    3. PitchVisionModel (sliding 5-frame window): off-ball intent predictions.
    4. MatchAccumulator: roll everything up into per-player profiles + event log.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from pitchvision.data.event_detector import EventDetector
from pitchvision.inference.accumulator import MatchAccumulator
from pitchvision.models.pitchvision import PitchVisionModel
from pitchvision.video.pipeline import VideoProcessor

INTENT_LABELS = ["support", "press", "space", "hold", "unknown"]


class MatchProcessor:
    def __init__(self, cfg, model_ckpt: str = None, homography_path: str = None,
                 device: str = None):
        self.cfg = cfg
        self.num_frames = int(cfg.model.num_frames)
        self.num_players = int(cfg.model.num_players)
        self.kin_dim = int(cfg.token.kinematic_dim)
        self.field_length = float(cfg.data.field_length)
        self.field_width = float(cfg.data.field_width)
        self.tracking_hz = int(cfg.data.tracking_hz)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.video = VideoProcessor(homography_path=homography_path,
                                    device=str(self.device))
        self.event_detector = EventDetector(tracking_hz=self.tracking_hz)
        self.accumulator = MatchAccumulator()

        self.model = PitchVisionModel(cfg).to(self.device).eval()
        if model_ckpt:
            ckpt = torch.load(model_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])

    # --------------------------------------------------- top-level
    def process(self, video_path: str, output_dir: str,
                max_frames: int = None) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tracking_path = str(out_dir / "tracking.json")
        self.video.process_video(video_path, tracking_path,
                                 max_frames=max_frames)

        with open(tracking_path) as f:
            tracking_data = json.load(f)
        frames = tracking_data["frames"]

        events = self.event_detector.detect_events(frames)
        self.accumulator.add_events(events)

        self._predict_intents(frames)
        self._accumulate_kinematics(frames)

        profiles = self.accumulator.finalize()
        report = {
            "player_profiles": {pid: vars(p) for pid, p in profiles.items()},
            "event_log": self.accumulator.event_log(),
            "match_summary": {
                "total_frames": len(frames),
                "total_events": len(self.accumulator.events),
                "players_tracked": len(profiles),
            },
        }

        report_path = out_dir / "scouting_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report: {report_path}")
        return report

    # --------------------------------------------------- inner loops
    @torch.no_grad()
    def _predict_intents(self, frames: list):
        buf = deque(maxlen=self.num_frames)
        for frame in frames:
            buf.append(frame)
            if len(buf) < self.num_frames:
                continue
            batch = self._build_batch(list(buf))
            pred = self.model(batch)
            intent_idx = pred["intent_logits"][0].argmax(dim=-1)     # (N,)
            urgency = pred["urgency"][0]                              # (N,)

            players = frame.get("players", [])[:self.num_players]
            for pi, player in enumerate(players):
                label = INTENT_LABELS[int(intent_idx[pi])]
                self.accumulator.add_intent(
                    player["track_id"], label, float(urgency[pi]))

    def _accumulate_kinematics(self, frames: list):
        dt = 1.0 / self.tracking_hz
        prev: dict = {}
        for frame in frames:
            for player in frame.get("players", []):
                pid = player["track_id"]
                pos = np.array(player["pos"], dtype=np.float32)
                if pid in prev:
                    speed = float(np.linalg.norm(pos - prev[pid]) / dt)
                    self.accumulator.add_kinematics(pid, speed, dt)
                prev[pid] = pos

    # --------------------------------------------------- batch assembly
    def _build_batch(self, frames: list) -> dict:
        T = self.num_frames
        N = self.num_players
        K = self.kin_dim

        pos_seq = torch.zeros(1, T, N + 1, 2, device=self.device)
        kin_seq = torch.zeros(1, T, N, K, device=self.device)
        role_seq = torch.zeros(1, T, N, dtype=torch.long, device=self.device)
        valid_mask = torch.zeros(1, T, N + 1, dtype=torch.bool, device=self.device)

        prev_pos = [None] * N  # for per-player velocity estimation

        for fi, frame in enumerate(frames):
            ball = frame.get("ball")
            if ball is not None:
                pos_seq[0, fi, 0, 0] = ball["pos"][0] / (self.field_length / 2)
                pos_seq[0, fi, 0, 1] = ball["pos"][1] / (self.field_width / 2)
                valid_mask[0, fi, 0] = True

            for pi, player in enumerate(frame.get("players", [])[:N]):
                px = player["pos"][0] / (self.field_length / 2)
                py = player["pos"][1] / (self.field_width / 2)
                pos_seq[0, fi, pi + 1, 0] = px
                pos_seq[0, fi, pi + 1, 1] = py
                valid_mask[0, fi, pi + 1] = True

                if prev_pos[pi] is not None:
                    vx = (player["pos"][0] - prev_pos[pi][0]) * self.tracking_hz
                    vy = (player["pos"][1] - prev_pos[pi][1]) * self.tracking_hz
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    kin_seq[0, fi, pi, 0] = vx
                    kin_seq[0, fi, pi, 1] = vy
                    if K >= 3:
                        kin_seq[0, fi, pi, 2] = speed
                prev_pos[pi] = list(player["pos"])

        return {
            "pos_seq": pos_seq,
            "kin_seq": kin_seq,
            "role_seq": role_seq,
            "valid_mask": valid_mask,
        }
