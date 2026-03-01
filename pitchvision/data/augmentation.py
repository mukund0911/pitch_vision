"""
Augmentations in field-coordinate space.

These work on the 5-frame episodic format used at training time. They are
geometric, not pixel-level, so they stay consistent with the coordinates.
"""

import random
from copy import deepcopy

import numpy as np


class TacticalAugmentation:
    def __init__(self, mirror_p: float = 0.5, velocity_noise: float = 0.5,
                 stretch_range: tuple = (0.88, 1.12)):
        self.mirror_p = mirror_p
        self.velocity_noise = velocity_noise
        self.stretch_range = stretch_range

    def __call__(self, frames: list) -> list:
        frames = deepcopy(frames)
        if random.random() < self.mirror_p:
            self._mirror(frames)
        self._velocity_noise(frames)
        self._stretch(frames)
        return frames

    def _mirror(self, frames):
        # Flip x; teams swap which half they attack. Doesn't change intent labels.
        for f in frames:
            if f.get("ball") is not None:
                f["ball"]["pos"][0] = -f["ball"]["pos"][0]
            for p in f.get("players", []):
                p["pos"][0] = -p["pos"][0]
                if "velocity" in p:
                    p["velocity"][0] = -p["velocity"][0]

    def _velocity_noise(self, frames):
        if self.velocity_noise <= 0:
            return
        for f in frames:
            for p in f.get("players", []):
                if "velocity" not in p:
                    continue
                noise = np.random.normal(0, self.velocity_noise, size=2)
                p["velocity"][0] = float(p["velocity"][0] + noise[0])
                p["velocity"][1] = float(p["velocity"][1] + noise[1])

    def _stretch(self, frames):
        scale = random.uniform(*self.stretch_range)
        for f in frames:
            if f.get("ball") is not None:
                f["ball"]["pos"] = [c * scale for c in f["ball"]["pos"]]
            for p in f.get("players", []):
                p["pos"] = [c * scale for c in p["pos"]]
