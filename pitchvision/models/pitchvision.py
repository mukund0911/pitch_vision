"""
PitchVisionModel — ties the token encoder, transformer, decoder, and rollout
into a single forward pass.

Input batch (keys):
    pos_seq    (B, T, N+1, 2)   ball at index 0, then N players
    kin_seq    (B, T, N, 4)     players only
    role_seq   (B, T, N)        players only, long
    valid_mask (B, T, N+1)      bool

Sequence layout (frame-major):
    [frame0: ball, p1, p2, ..., pN,  frame1: ball, p1, ..., pN,  ...]
    token_count = T * (N+1)

Output (keys):
    dir_logits    (B, N, num_directions)
    intent_logits (B, N, num_intents)
    urgency       (B, N)
    next_pos      (B, N, 2)
    attn_rollout  (B, N)
"""

import torch
import torch.nn as nn

from .attention import TacticalEncoder
from .decoder import IntentDecoder
from .encodings import FourierPositionalEncoding, FrameEmbedding
from .rollout import AttentionRollout
from .token import PlayerToken


class PitchVisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"] if isinstance(cfg, dict) else cfg.model
        tok = cfg["token"] if isinstance(cfg, dict) else cfg.token

        self.d_model = int(m["d_model"])
        self.num_frames = int(m["num_frames"])
        self.num_players = int(m["num_players"])
        self.num_directions = int(m["num_directions"])
        self.num_intents = int(m["num_intents"])
        self.fourier_bands = int(tok["fourier_bands"])
        self.kinematic_dim = int(tok["kinematic_dim"])

        self.player_token = PlayerToken(
            d_model=self.d_model,
            num_roles=int(m["num_roles"]),
            kinematic_dim=self.kinematic_dim,
            fourier_bands=self.fourier_bands,
        )
        # Ball has no role/kinematics in the player sense — we encode only its
        # position and add a learned "ball" type vector.
        self.ball_pos_enc = FourierPositionalEncoding(
            in_dim=2, out_dim=self.d_model, num_bands=self.fourier_bands)
        self.ball_type = nn.Parameter(torch.randn(self.d_model) * 0.02)

        self.frame_emb = FrameEmbedding(
            num_frames=self.num_frames, d_model=self.d_model)

        self.encoder = TacticalEncoder(
            d_model=self.d_model,
            num_heads=int(m["num_heads"]),
            ffn_dim=int(m["ffn_dim"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
        )

        self.decoder = IntentDecoder(
            d_model=self.d_model,
            num_directions=self.num_directions,
            num_intents=self.num_intents,
        )

        self.rollout = AttentionRollout()

    def forward(self, batch: dict) -> dict:
        pos_seq: torch.Tensor = batch["pos_seq"]       # (B, T, N+1, 2)
        kin_seq: torch.Tensor = batch["kin_seq"]       # (B, T, N, 4)
        role_seq: torch.Tensor = batch["role_seq"]     # (B, T, N)
        valid_mask: torch.Tensor = batch["valid_mask"] # (B, T, N+1) bool

        B, T, N_plus_1, _ = pos_seq.shape
        N = N_plus_1 - 1
        assert T == self.num_frames
        assert N == self.num_players

        # --- Tokenize ---
        ball_pos = pos_seq[:, :, 0, :]                 # (B, T, 2)
        player_pos = pos_seq[:, :, 1:, :]              # (B, T, N, 2)

        ball_tokens = self.ball_pos_enc(ball_pos) + self.ball_type  # (B, T, d)
        ball_tokens = ball_tokens.unsqueeze(2)                       # (B, T, 1, d)

        player_tokens = self.player_token(
            player_pos, kin_seq, role_seq)              # (B, T, N, d)

        # (B, T, N+1, d) with ball at index 0
        tokens = torch.cat([ball_tokens, player_tokens], dim=2)

        # --- Add frame embeddings (broadcast over entities in a frame) ---
        frame_idx = torch.arange(T, device=tokens.device)
        frame_e = self.frame_emb(frame_idx)             # (T, d)
        tokens = tokens + frame_e.view(1, T, 1, self.d_model)

        # --- Flatten to (B, T*(N+1), d) ---
        seq = tokens.view(B, T * N_plus_1, self.d_model)
        seq_mask = valid_mask.view(B, T * N_plus_1)

        # --- Encode ---
        encoded, attn_layers = self.encoder(seq, mask=seq_mask)

        # --- Pull out final-frame player tokens ---
        # Final frame starts at offset (T-1)*(N+1). Ball is at that offset, then
        # players are the next N tokens.
        final_frame_offset = (T - 1) * N_plus_1
        final_ball_idx = final_frame_offset
        final_player_idxs = list(range(final_frame_offset + 1,
                                       final_frame_offset + 1 + N))

        final_player_tokens = encoded[:, final_player_idxs, :]   # (B, N, d)
        decoded = self.decoder(final_player_tokens)

        # --- Attention rollout: ball -> players (final frame) ---
        rollout_full = self.rollout(attn_layers)                 # (B, N_tot, N_tot)
        influence = AttentionRollout.ball_to_players(
            rollout_full, final_ball_idx, final_player_idxs)     # (B, N)

        decoded["attn_rollout"] = influence
        return decoded

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
