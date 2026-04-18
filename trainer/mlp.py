import torch
import torch.nn as nn

import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, num_freq=8):
        super().__init__()
        self.num_freq = num_freq
        self.out_dim = 2 * 2 * num_freq

    def forward(self, uv: torch.Tensor):
        """
        uv: [B, 2]
        return: [B, out_dim]
        """
        freq_bands = 2.0 ** torch.arange(self.num_freq, device=uv.device).float()
        uv_scaled = uv.unsqueeze(-1) * freq_bands * np.pi

        return torch.cat([
            torch.sin(uv_scaled).reshape(uv.shape[0], -1),
            torch.cos(uv_scaled).reshape(uv.shape[0], -1)
        ], dim=-1)


class MlpDecoder(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int, output_channels: int, hidden_dim=64, num_hidden=2):
        super().__init__()

        input_dim = latent_dim + pe_dim + 1

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

        layers.append(nn.Linear(hidden_dim, output_channels))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, latent_features: torch.Tensor, pe_features: torch.Tensor, mip_level):
        """
        latent_features: [B, latent_dim]
        pe_features: [B, pe_dim]
        mip_level: [B, 1]
        """
        x = torch.cat([latent_features, pe_features, mip_level], dim=-1)
        return self.net(x)
