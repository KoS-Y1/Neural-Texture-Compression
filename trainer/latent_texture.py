import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np


# Load PBR texture bundle
def load_pbr_texture(paths: dict, resolution: int | None = None) -> torch.Tensor:
    """
      paths: {
          'albedo': 'albedo.png',      # 3 channels
          'normal': 'normal.png',      # 3 channels
          'roughness': 'roughness.png', # 1 channel
          'metalness': 'metalness.png', # 1 channel
          'ao': 'ao.png',              # 1 channel
      }

      resolution: if None, use the native size of the first image; all others are
      resized to match so channels stack cleanly.

      returns: tensor [C, H, W] with all channels stacked, values in [0, 1]
      """
    channels = []
    target_size = None
    for name, path in paths.items():
        img = Image.open(path)

        if target_size is None:
            target_size = (resolution, resolution) if resolution is not None else img.size

        if img.size != target_size:
            img = img.resize(target_size)

        # Load image and convert it to float value [H, W, C]
        t = torch.from_numpy(np.array(img)).float() / 255.0

        # If single channel (roughness, metallic, etc.)
        # [H, W] to [H, W, C]
        if t.ndim == 2:
            t = t.unsqueeze(-1)

        channels.append(t)
    bundle = torch.cat(channels, dim=-1)
    return bundle.permute(2, 0, 1)  # [C_total, H, W]


# Generate mipmaps from input texture
def gen_mipmaps(texture: torch.Tensor, num_levels=None) -> list[torch.Tensor]:
    if num_levels is None:
        num_levels = int(np.log2(min(texture.shape[1], texture.shape[2])))

    mipmaps = [texture]

    for _ in range(num_levels - 1):
        prev = mipmaps[-1]
        downsampled = F.avg_pool2d(prev.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
        mipmaps.append(downsampled)

    return mipmaps

def quantize_ste(x, num_bits=8):
    """Quantization with straight-through estimator (used during training)."""
    qmax = (2 ** num_bits) - 1

    x_clamped = torch.clamp(x, 0, 1)
    x_quant = torch.round(x_clamped * qmax) / qmax

    return x_clamped + (x_quant - x_clamped).detach()


class LatentTexture(nn.Module):
    def __init__(self, hi_res: int, hi_channels: int, low_res: int, lo_channels: int):
        super().__init__()
        # Initialize in [0, 1] so quantize_ste's clamp is a no-op at step 0 and the full
        # latent grid contributes from the start rather than half the cells starting at 0.
        self.latent_hi = nn.Parameter(torch.rand(1, hi_channels, hi_res, hi_res))
        self.latent_lo = nn.Parameter(torch.rand(1, lo_channels, low_res, low_res))

    def sample(self, uv: torch.Tensor, hi_bits=8, lo_bits=4) -> torch.Tensor:
        """
         uv: [B, 2] in range [0, 1]
         return: [B, hi_channels + lo_channels]
        """
        # Remap to [-1, 1]; pack B points into the output-H dim of a single-batch grid
        grid = (uv * 2 - 1).view(1, -1, 1, 2)  # [1, B, 1, 2]
        hi_q = quantize_ste(self.latent_hi, hi_bits)
        lo_q = quantize_ste(self.latent_lo, lo_bits)

        hi = F.grid_sample(
            hi_q,
            grid,
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(-1).transpose(0, 1)  # [B, hi_channels]

        lo = F.grid_sample(
            lo_q,
            grid,
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(-1).transpose(0, 1)  # [B, lo_channels]

        return torch.cat([hi, lo], dim=-1)
