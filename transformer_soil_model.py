"""Transformer-based multimodal soil-moisture model.

The model keeps the CNN_Spain input contract:

* ``x`` has shape ``[batch, channels, 13, 13]``.
* ``seq`` has shape ``[batch, sequence, temporal_features]``.
* The output is a residual prediction with shape ``[batch, 1]``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TransformerSoilNet(nn.Module):
    """Augment the existing CNN/LSTM soil model with transformer features.

    The original spatial CNN, static MLP, and temporal LSTM remain active.
    Transformer embeddings are learned in parallel and concatenated before
    the original fusion head.
    """

    def __init__(
        self,
        in_channels: int,
        dynamic_idx: list[int],
        static_idx: list[int],
        seq_input_size: int = 4,
        *,
        patch_size: int = 13,
        d_model: int = 128,
        nhead: int = 8,
        spatial_layers: int = 3,
        temporal_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if not dynamic_idx:
            raise ValueError("dynamic_idx must contain at least one channel")
        if not static_idx:
            raise ValueError("static_idx must contain at least one channel")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.patch_size = patch_size
        self.register_buffer("dynamic_idx", torch.tensor(dynamic_idx, dtype=torch.long))
        self.register_buffer("static_idx", torch.tensor(static_idx, dtype=torch.long))

        spatial_input_size = len(dynamic_idx) + len(static_idx)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(len(dynamic_idx), 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(len(static_idx), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.temporal_lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.spatial_projection = nn.Linear(spatial_input_size, d_model)
        self.spatial_position = nn.Parameter(torch.zeros(1, patch_size * patch_size, d_model))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)
        self.spatial_norm = nn.LayerNorm(d_model)

        self.temporal_projection = nn.Linear(seq_input_size, d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Original CNN/LSTM features plus transformer features and routed
        # center pixels.
        fusion_size = 128 * 3 * 3 + 32 + 96 + 2 * d_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: Tensor, seq: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [batch, channels, height, width], got {tuple(x.shape)}")
        if seq.ndim != 3:
            raise ValueError(f"seq must have shape [batch, time, features], got {tuple(seq.shape)}")
        if x.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"x patch must be {self.patch_size}x{self.patch_size}, got {tuple(x.shape[-2:])}"
            )

        dynamic = x.index_select(1, self.dynamic_idx)
        static = x.index_select(1, self.static_idx)
        center_dynamic = dynamic[:, :, self.patch_size // 2, self.patch_size // 2]
        center_static = static[:, :, self.patch_size // 2, self.patch_size // 2]
        cnn_embedding = self.spatial_cnn(dynamic)
        static_embedding = self.static_mlp(center_static)
        _, (hidden, _) = self.temporal_lstm(seq)
        lstm_embedding = hidden[-1]

        spatial = torch.cat([dynamic, static], dim=1).flatten(2).transpose(1, 2)
        spatial = self.spatial_projection(spatial) + self.spatial_position
        spatial = self.spatial_norm(self.spatial_encoder(spatial))
        spatial_embedding = spatial.mean(dim=1)

        temporal = self.temporal_projection(seq)
        temporal = self.temporal_norm(self.temporal_encoder(temporal))
        temporal_embedding = temporal[:, -1]

        center = x[:, :, self.patch_size // 2, self.patch_size // 2]
        fused = torch.cat(
            [
                cnn_embedding,
                static_embedding,
                lstm_embedding,
                spatial_embedding,
                temporal_embedding,
            ],
            dim=1,
        )
        return self.fusion(fused)
