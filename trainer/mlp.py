import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Asymmetric quantization range: [-(N-1)/2 * Q, N/2 * Q] with N = 2^B and Q = 1/N.
DEFAULT_GRID_BITS = 4


def quant_step(num_bits: int) -> float:
    return 1.0 / float(1 << num_bits)


def quant_range(num_bits: int) -> tuple[float, float]:
    n = 1 << num_bits
    q = quant_step(num_bits)
    return -((n - 1) * 0.5) * q, (n * 0.5) * q


def quantize_to_codes(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    n = 1 << num_bits
    q = quant_step(num_bits)
    lo, _ = quant_range(num_bits)
    codes = torch.round((x - lo) / q).clamp(0, n - 1)
    return codes.to(torch.uint8 if num_bits <= 8 else torch.int16)


def dequantize_from_codes(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    q = quant_step(num_bits)
    lo, _ = quant_range(num_bits)
    return codes.float() * q + lo


def add_quant_noise(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    q = quant_step(num_bits)
    return x + (torch.rand_like(x) - 0.5) * q


class FeatureLevel(nn.Module):
    """One pyramid level: high-res G0 (4-neighbor concat) + low-res G1 (bilinear)."""

    def __init__(self, c0: int, res0: int, c1: int, res1: int, num_bits: int = DEFAULT_GRID_BITS):
        super().__init__()
        self.c0, self.c1 = c0, c1
        self.res0, self.res1 = res0, res1
        self.num_bits = num_bits
        lo, hi = quant_range(num_bits)
        scale = (hi - lo) * 0.25
        self.g0 = nn.Parameter(torch.empty(1, c0, res0, res0).uniform_(-scale, scale))
        self.g1 = nn.Parameter(torch.empty(1, c1, res1, res1).uniform_(-scale, scale))

    def _quant(self, x: torch.Tensor, qat_noise: bool, hard: bool) -> torch.Tensor:
        if hard:
            codes = quantize_to_codes(x.detach(), self.num_bits)
            return dequantize_from_codes(codes, self.num_bits).to(x.dtype).to(x.device)
        if qat_noise and self.training:
            return add_quant_noise(x, self.num_bits)
        return x

    @torch.no_grad()
    def clamp_to_quant_range(self) -> None:
        lo, hi = quant_range(self.num_bits)
        self.g0.clamp_(lo, hi)
        self.g1.clamp_(lo, hi)

    def sample(self, uv: torch.Tensor, qat_noise: bool = True, hard_quant: bool = False) -> torch.Tensor:
        g0 = self._quant(self.g0, qat_noise, hard_quant)
        g1 = self._quant(self.g1, qat_noise, hard_quant)

        # G0: 4 nearest neighbors concatenated; the MLP learns the blend.
        pos = uv * self.res0 - 0.5
        x0 = torch.floor(pos[:, 0]).long().clamp(0, self.res0 - 2)
        y0 = torch.floor(pos[:, 1]).long().clamp(0, self.res0 - 2)
        x1, y1 = x0 + 1, y0 + 1
        plane = g0[0].permute(1, 2, 0)  # [H, W, C0]
        feat0 = torch.cat([plane[y0, x0], plane[y0, x1], plane[y1, x0], plane[y1, x1]], dim=-1)

        # G1: bilinear sample.
        grid = (uv * 2.0 - 1.0).view(1, -1, 1, 2)
        feat1 = F.grid_sample(g1, grid, mode='bilinear', align_corners=False)
        feat1 = feat1.squeeze(0).squeeze(-1).transpose(0, 1)  # [B, C1]

        return torch.cat([feat0, feat1], dim=-1)


class LatentPyramid(nn.Module):

    def __init__(self, base_resolution: int, c0: int = 12, c1: int = 20, num_bits: int = DEFAULT_GRID_BITS):
        super().__init__()
        self.c0, self.c1 = c0, c1
        self.num_bits = num_bits
        self.base_resolution = base_resolution

        max_mip = int(math.log2(base_resolution))
        meta: list[tuple[int, int, int, int]] = []
        cursor, cur_res = 0, base_resolution
        while cursor <= max_mip:
            span = min(4 if not meta else 2, max_mip - cursor + 1)
            res0 = max(cur_res // 4, 2)
            res1 = max(cur_res // 8, 2)
            meta.append((cursor, cursor + span - 1, res0, res1))
            cursor += span
            cur_res = max(cur_res // (1 << span), 1)
        if meta and meta[-1][1] < max_mip:
            top, _, r0, r1 = meta[-1]
            meta[-1] = (top, max_mip, r0, r1)

        self._meta = meta
        self.levels = nn.ModuleList([
            FeatureLevel(c0, r0, c1, r1, num_bits) for (_, _, r0, r1) in meta
        ])

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def level_for_mip(self, mip: int) -> int:
        for i, (top, bot, _, _) in enumerate(self._meta):
            if top <= mip <= bot:
                return i
        return self.num_levels - 1

    def level_meta(self, idx: int) -> dict:
        top, bot, r0, r1 = self._meta[idx]
        return {"top_mip": top, "bottom_mip": bot, "res0": r0, "res1": r1, "c0": self.c0, "c1": self.c1}

    def sample(self, uv: torch.Tensor, mip: int, qat_noise: bool = True, hard_quant: bool = False) -> torch.Tensor:
        return self.levels[self.level_for_mip(mip)].sample(uv, qat_noise, hard_quant)

    @torch.no_grad()
    def clamp_to_quant_range(self) -> None:
        for level in self.levels:
            level.clamp_to_quant_range()


class TiledPositionalEncoder(nn.Module):
    """Triangular-wave tiled PE (paper Sec 4.3.2).

    Pattern repeats every `tile` texels at the highest mip. `octaves` octaves
    per axis (frequencies 1, 2, 4, ...), sin and cos pair per octave; for
    tile=8 and octaves=3 (= log2 8) the output is 12 scalars.
    """

    def __init__(self, tile: int = 8, octaves: int = 3):
        super().__init__()
        self.tile = tile
        self.num_freq = octaves
        self.out_dim = 2 * 2 * octaves

    @staticmethod
    def _tri_sin(t: torch.Tensor) -> torch.Tensor:
        f = (t + 0.25) - torch.floor(t + 0.25)
        return 1.0 - 4.0 * torch.abs(f - 0.5)

    @staticmethod
    def _tri_cos(t: torch.Tensor) -> torch.Tensor:
        f = (t + 0.5) - torch.floor(t + 0.5)
        return 1.0 - 4.0 * torch.abs(f - 0.5)

    def forward(self, uv: torch.Tensor, base_resolution: int) -> torch.Tensor:
        # One frequency-1 cycle == one tile at the texture's top resolution.
        cycles = uv * (base_resolution / float(self.tile))
        outs: list[torch.Tensor] = []
        for axis in range(2):
            t = cycles[:, axis:axis + 1]
            for k in range(self.num_freq):
                phase = t * float(1 << k)
                outs.append(self._tri_sin(phase))
                outs.append(self._tri_cos(phase))
        return torch.cat(outs, dim=-1)


# From NTC paper Sec 4.4: cheap approximation of GELU used at inference.
class HardGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < -1.5, torch.zeros_like(x),
                           torch.where(x > 1.5, x, x / 3.0 * (x + 1.5)))


class MlpDecoder(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int, output_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = latent_dim + pe_dim + 1  # +1 for normalized LOD
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            HardGELU(),
            nn.Linear(hidden_dim, hidden_dim),
            HardGELU(),
            nn.Linear(hidden_dim, output_channels),
        )

    def forward(self, latent_features: torch.Tensor, pe_features: torch.Tensor,
                mip_level: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_features, pe_features, mip_level], dim=-1))
