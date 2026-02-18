"""
SoccerIntentDataset — PyTorch Dataset for training the tactical transformer.

Each sample is a 5-frame window around a single player's intent episode.
Episodes are expected to be pre-built JSON files with structure:

    {
        "frames": [
            {
                "ball": {"pos": [x, y]},
                "players": [
                    {"pos": [x, y], "velocity": [vx, vy],
                     "speed": s, "accel": a, "role": "CM"},
                    ...
                ],
            },
            ...
        ],
        "target_player_idx": int,                 # which player to predict for
        "labels": {
            "direction": int,                     # 0-7
            "intent": int,                        # 0-4
            "urgency": float,                     # 0-1
            "next_pos": [dx, dy],                 # meters
            "causal_trigger_idx": int or null,
        },
    }
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROLE_MAP = {"GK": 0, "CB": 1, "FB": 2, "CM": 3, "AM": 4, "W": 5, "ST": 6}


class SoccerIntentDataset(Dataset):
    def __init__(self, episodes_dir: str, split: str = "train",
                 augment=None, num_frames: int = 5, num_players: int = 22,
                 kinematic_dim: int = 4):
        self.episodes_dir = Path(episodes_dir)
        self.augment = augment
        self.num_frames = num_frames
        self.num_players = num_players
        self.kinematic_dim = kinematic_dim

        split_file = self.episodes_dir.parent / "splits" / f"{split}.json"
        with open(split_file) as f:
            self.ids = json.load(f)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        with open(self.episodes_dir / f"{self.ids[idx]}.json") as f:
            ep = json.load(f)

        frames = ep["frames"][:self.num_frames]
        if self.augment is not None:
            frames = self.augment(frames)

        T, N, K = self.num_frames, self.num_players, self.kinematic_dim
        pos_seq = torch.zeros(T, N + 1, 2)
        kin_seq = torch.zeros(T, N, K)
        role_seq = torch.zeros(T, N, dtype=torch.long)
        valid_mask = torch.zeros(T, N + 1, dtype=torch.bool)

        for fi, frame in enumerate(frames):
            ball = frame.get("ball")
            if ball is not None:
                pos_seq[fi, 0] = torch.tensor(ball["pos"], dtype=torch.float32)
                valid_mask[fi, 0] = True

            for pi, player in enumerate(frame.get("players", [])[:N]):
                pos_seq[fi, pi + 1] = torch.tensor(player["pos"], dtype=torch.float32)
                kin_seq[fi, pi] = self._kin_vec(player)
                role_seq[fi, pi] = ROLE_MAP.get(player.get("role", "CM"), 3)
                valid_mask[fi, pi + 1] = True

        lbl = ep["labels"]

        causal = torch.zeros(N)
        trig = lbl.get("causal_trigger_idx")
        if trig is not None and 0 <= trig < N:
            causal[trig] = 1.0
        causal = torch.softmax(causal / 2.0, dim=0)

        return {
            "pos_seq": pos_seq,
            "kin_seq": kin_seq,
            "role_seq": role_seq,
            "valid_mask": valid_mask,
            "target_idx": int(ep["target_player_idx"]),
            "direction": torch.tensor(lbl["direction"], dtype=torch.long),
            "intent": torch.tensor(lbl["intent"], dtype=torch.long),
            "urgency": torch.tensor(lbl["urgency"], dtype=torch.float32),
            "next_pos": torch.tensor(lbl["next_pos"], dtype=torch.float32),
            "causal_dist": causal,
        }

    def _kin_vec(self, player: dict) -> torch.Tensor:
        vx = float(player.get("velocity", [0.0, 0.0])[0])
        vy = float(player.get("velocity", [0.0, 0.0])[1])
        speed = float(player.get("speed", (vx ** 2 + vy ** 2) ** 0.5))
        accel = float(player.get("accel", 0.0))
        if self.kinematic_dim == 4:
            return torch.tensor([vx, vy, speed, accel], dtype=torch.float32)
        return torch.tensor([vx, vy][:self.kinematic_dim], dtype=torch.float32)
