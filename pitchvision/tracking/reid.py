"""
Appearance embedding network for player re-identification.

Reference: Wojke et al. (2017), Section 2.3.

A small CNN that maps a 128x64 player crop to a 128-dim unit vector. L2
normalization on the output means cosine similarity is just a dot product.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class ReIDNet(nn.Module):
    INPUT_H = 128
    INPUT_W = 64

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            _conv_block(3, 32, stride=2),     # 64 x 32
            _conv_block(32, 64, stride=2),    # 32 x 16
            _conv_block(64, 128, stride=2),   # 16 x 8
            _conv_block(128, 256, stride=2),  #  8 x 4
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def extract(self, crops: list) -> np.ndarray:
        """
        Compute embeddings for a list of BGR uint8 crops.
        """
        if len(crops) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        device = next(self.parameters()).device
        batch = np.empty((len(crops), 3, self.INPUT_H, self.INPUT_W), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.size == 0:
                batch[i] = 0.0
                continue
            img = cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch[i] = img.transpose(2, 0, 1)

        tensor = torch.from_numpy(batch).to(device)
        embeddings = self.forward(tensor)
        return embeddings.cpu().numpy()
