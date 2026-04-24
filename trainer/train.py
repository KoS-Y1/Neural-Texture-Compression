import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mlp

PROJECT_ROOT = _THIS_DIR.parent
ASSETS_LOAD_DIR = PROJECT_ROOT / "assets/source"
ASSETS_EXPORT_DIR = PROJECT_ROOT / "assets/export"


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


def _sample_lod(num_mip_levels: int, device: torch.device) -> int:
    """Paper Sec 5.1: LOD = floor(-log_4 X), X~U(0,1); 5% uniform fallback."""
    if torch.rand((), device=device).item() < 0.05:
        return int(torch.randint(0, num_mip_levels, (1,), device=device).item())
    x = torch.rand((), device=device).clamp_min(1e-8).item()
    lod = int(math.floor(-math.log(x) / math.log(4)))
    return min(lod, num_mip_levels - 1)


def train_ntc(texture_bundle: torch.Tensor, num_iter, batch_size=65536, lr_latent=0.01, lr_mlp=0.005,
              c0=12, c1=20, num_bits=4, qat_freeze_frac=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = texture_bundle.shape[0]
    base_resolution = texture_bundle.shape[-1]

    mips = gen_mipmaps(texture_bundle)
    num_mip_levels = len(mips)
    mips = [m.to(device) for m in mips]

    pyramid = mlp.LatentPyramid(base_resolution, c0=c0, c1=c1, num_bits=num_bits).to(device)
    pe = mlp.TiledPositionalEncoder().to(device)
    mlp_decoder = mlp.MlpDecoder(
        latent_dim=c0 * 4 + c1,
        pe_dim=pe.out_dim,
        output_channels=num_channels,
        hidden_dim=64,
    ).to(device)

    print(f"Pyramid: {pyramid.num_levels} levels, c0={c0}, c1={c1}, {num_bits}-bit grids")
    for i in range(pyramid.num_levels):
        m = pyramid.level_meta(i)
        print(f"  L{i}: G0={m['res0']}^2 G1={m['res1']}^2  mips {m['top_mip']}..{m['bottom_mip']}")
    print(f"MLP input dim: {mlp_decoder.input_dim} (latent {c0 * 4 + c1} + pe {pe.out_dim} + lod 1)")

    optimizer = torch.optim.Adam([
        {'params': pyramid.parameters(), 'lr': lr_latent},
        {'params': mlp_decoder.parameters(), 'lr': lr_mlp},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=0)

    freeze_step = int(num_iter * (1.0 - qat_freeze_frac))
    grids_frozen = False

    for step in range(num_iter):
        mip_idx = _sample_lod(num_mip_levels, device)
        mip_texture = mips[mip_idx]
        H_m, W_m = mip_texture.shape[1], mip_texture.shape[2]
        n = min(batch_size, H_m * W_m)

        px = torch.randint(0, W_m, (n,), device=device)
        py = torch.randint(0, H_m, (n,), device=device)
        u = (px.float() + 0.5) / W_m
        v = (py.float() + 0.5) / H_m
        uv = torch.stack([u, v], dim=-1)
        ground_truth = mip_texture[:, py, px].T

        # QAT freeze phase: hard-quantize grids and stop updating them; only fine-tune the MLP.
        if step == freeze_step and not grids_frozen:
            pyramid.clamp_to_quant_range()
            for p in pyramid.parameters():
                p.requires_grad_(False)
            grids_frozen = True
            print(f"Step {step}: freezing latent grids, fine-tuning MLP only for last {num_iter - step} steps")

        latent_feature = pyramid.sample(uv, mip_idx, qat_noise=not grids_frozen, hard_quant=grids_frozen)
        pe_feature = pe(uv, base_resolution)
        mip_norm = torch.full((n, 1), mip_idx / max(num_mip_levels - 1, 1), device=device)
        pred = mlp_decoder(latent_feature, pe_feature, mip_norm)

        loss = F.mse_loss(pred, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not grids_frozen:
            pyramid.clamp_to_quant_range()

        if step % 500 == 0 or step == num_iter - 1:
            psnr = -10 * torch.log10(loss).item()
            print(f"Step {step} / {num_iter} | mip {mip_idx} | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")

    return pyramid, mlp_decoder, pe


def export_ntc(pyramid: mlp.LatentPyramid, mlp_decoder: mlp.MlpDecoder, pe: mlp.TiledPositionalEncoder, path):
    torch.save({
        'pyramid_state': pyramid.state_dict(),
        'pyramid_meta': [pyramid.level_meta(i) for i in range(pyramid.num_levels)],
        'num_bits': pyramid.num_bits,
        'mlp_weights': mlp_decoder.state_dict(),
        'pe_tile': pe.tile,
        'pe_octaves': pe.num_freq,
        'config': {
            'base_resolution': pyramid.base_resolution,
            'c0': pyramid.c0,
            'c1': pyramid.c1,
        }
    }, path)


def load_ntc(path: str | Path, device: torch.device | None = None) -> tuple[
    mlp.LatentPyramid, mlp.MlpDecoder, mlp.TiledPositionalEncoder]:
    """
    Loads a saved NTC model checkpoint and reconstructs the network architectures.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint dictionary
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Extract architecture configuration
    config = checkpoint['config']
    base_resolution = config['base_resolution']
    c0 = config['c0']
    c1 = config['c1']
    num_bits = checkpoint['num_bits']

    pe = mlp.TiledPositionalEncoder().to(device)
    # Restore the attributes if they are mutable in your mlp implementation
    if hasattr(pe, 'tile'):
        pe.tile = checkpoint['pe_tile']
    if hasattr(pe, 'num_freq'):
        pe.num_freq = checkpoint['pe_octaves']

    pyramid = mlp.LatentPyramid(
        base_resolution=base_resolution,
        c0=c0,
        c1=c1,
        num_bits=num_bits
    ).to(device)
    pyramid.load_state_dict(checkpoint['pyramid_state'])

    mlp_state = checkpoint['mlp_weights']

    # Infer output channels by looking at the first dimension of the last weight tensor
    weight_keys = [k for k in mlp_state.keys() if 'weight' in k]
    last_weight_key = weight_keys[-1]
    output_channels = mlp_state[last_weight_key].shape[0]

    mlp_decoder = mlp.MlpDecoder(
        latent_dim=c0 * 4 + c1,
        pe_dim=pe.out_dim,
        output_channels=output_channels,
        hidden_dim=64
    ).to(device)
    mlp_decoder.load_state_dict(mlp_state)

    return pyramid, mlp_decoder, pe

def _grid_to_array_layers(grid: torch.Tensor, num_bits: int) -> np.ndarray:
    """[1, C, H, W] grid (in asymmetric quant range) -> [L, H, W, 4] UNORM8 layout.

    Hard-quantizes via mlp.quantize_to_codes and rescales codes to [0, 255] so
    the runtime can sample the texture as VK_FORMAT_R8G8B8A8_UNORM (and undo
    the rescale by multiplying with the quant range when reconstructing).
    """
    tensor = grid.detach()[0]
    c, h, w = tensor.shape
    assert c % 4 == 0, f"grid channels must be divisible by 4, got {c}"
    codes = mlp.quantize_to_codes(tensor, num_bits)  # [0, N-1]
    n = (1 << num_bits) - 1
    rescaled = (codes.float() / n * 255.0).round().to(torch.uint8).cpu().numpy()
    layered = rescaled.reshape(c // 4, 4, h, w).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(layered)


def export_runtime(
        pyramid: mlp.LatentPyramid,
        mlp_decoder: mlp.MlpDecoder,
        pe: mlp.TiledPositionalEncoder,
        out_dir: Path,
):
    """Export latent grids, MLP weights, and a JSON header for the C++ runtime.

    Layout:
      out_dir/
        ntc.json                   -- config, layer shapes, byte offsets, level table
        latent_hi.bin              -- level 0 G0, uint8 layer-major [L, H, W, 4] (back-compat)
        latent_lo.bin              -- level 0 G1, uint8 layer-major [L, H, W, 4] (back-compat)
        latent_hi_lvlN.bin (N>=1)  -- level N G0
        latent_lo_lvlN.bin (N>=1)  -- level N G1
        mlp.bin                    -- float16, weight [out, in] row-major then bias [out].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    levels_meta = []
    for i in range(pyramid.num_levels):
        level = pyramid.levels[i]
        meta = pyramid.level_meta(i)
        hi_buf = _grid_to_array_layers(level.g0, pyramid.num_bits)
        lo_buf = _grid_to_array_layers(level.g1, pyramid.num_bits)
        hi_name = "latent_hi.bin" if i == 0 else f"latent_hi_lvl{i}.bin"
        lo_name = "latent_lo.bin" if i == 0 else f"latent_lo_lvl{i}.bin"
        (out_dir / hi_name).write_bytes(hi_buf.tobytes())
        (out_dir / lo_name).write_bytes(lo_buf.tobytes())
        levels_meta.append({
            "index": i,
            "top_mip": meta["top_mip"],
            "bottom_mip": meta["bottom_mip"],
            "g0": {"file": hi_name, "width": meta["res0"], "height": meta["res0"], "channels": meta["c0"]},
            "g1": {"file": lo_name, "width": meta["res1"], "height": meta["res1"], "channels": meta["c1"]},
        })

    # Level 0 grids are also surfaced under the legacy keys for the existing single-level runtime.
    hi_meta = levels_meta[0]["g0"]
    lo_meta = levels_meta[0]["g1"]

    def _activation_name(m: nn.Module) -> str:
        if isinstance(m, nn.ReLU):
            return "relu"
        if isinstance(m, nn.Sigmoid):
            return "sigmoid"
        return "none"

    linear_layers: list[tuple[nn.Linear, str]] = []
    modules = list(mlp_decoder.net)
    for mi, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            activation = _activation_name(modules[mi + 1]) if mi + 1 < len(modules) else "none"
            linear_layers.append((m, activation))

    # Cooperative vector row-major matrices require each row to be a multiple of 16 bytes;
    # pad any FP16 layer whose input count isn't a multiple of 8 with zero columns.
    COOP_VEC_ROW_ALIGN_ELEMS = 8

    mlp_buffers: list[bytes] = []
    mlp_layers_meta = []
    offset = 0
    for linear, activation in linear_layers:
        weight = linear.weight.detach().cpu().contiguous().half().numpy()  # [out, in]
        bias = linear.bias.detach().cpu().contiguous().half().numpy()  # [out]

        in_dim = weight.shape[1]
        padded_in = ((in_dim + COOP_VEC_ROW_ALIGN_ELEMS - 1) // COOP_VEC_ROW_ALIGN_ELEMS) * COOP_VEC_ROW_ALIGN_ELEMS
        if padded_in != in_dim:
            weight = np.pad(weight, ((0, 0), (0, padded_in - in_dim)), mode="constant", constant_values=0).astype(
                np.float16)

        weight_bytes = weight.tobytes()
        bias_bytes = bias.tobytes()

        mlp_layers_meta.append({
            "in": int(weight.shape[1]),
            "out": int(weight.shape[0]),
            "activation": activation,
            "weight_offset": offset,
            "weight_size": len(weight_bytes),
            "bias_offset": offset + len(weight_bytes),
            "bias_size": len(bias_bytes),
        })

        mlp_buffers.append(weight_bytes)
        mlp_buffers.append(bias_bytes)
        offset += len(weight_bytes) + len(bias_bytes)

    (out_dir / "mlp.bin").write_bytes(b"".join(mlp_buffers))

    input_dim = mlp_layers_meta[0]["in"] if mlp_layers_meta else 0
    output_dim = mlp_layers_meta[-1]["out"] if mlp_layers_meta else 0
    quant_lo, quant_hi = mlp.quant_range(pyramid.num_bits)

    def _level0_compat(meta_g: dict) -> dict:
        return {
            "file": meta_g["file"],
            "width": meta_g["width"],
            "height": meta_g["height"],
            "channels": meta_g["channels"],
            "dtype": "uint8",
            "layout": "array_layers_hwc4",
            "sample_format": "unorm",
            "source_bits": pyramid.num_bits,
        }

    header = {
        "version": 2,
        "base_resolution": pyramid.base_resolution,
        "num_bits": pyramid.num_bits,
        "quant_range": [quant_lo, quant_hi],
        "latent_hi": _level0_compat(hi_meta),
        "latent_lo": _level0_compat(lo_meta),
        "levels": levels_meta,
        "positional_encoder": {
            "kind": "tiled_triangular",
            "tile": int(pe.tile),
            "octaves": int(pe.num_freq),
            "num_freq": int(pe.num_freq),
            "out_dim": int(pe.out_dim),
        },
        "mlp": {
            "file": "mlp.bin",
            "dtype": "float16",
            "weight_layout": "row_major_out_in",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "total_bytes": offset,
            "layers": mlp_layers_meta,
        },
    }

    with (out_dir / "ntc.json").open("w") as f:
        json.dump(header, f, indent=2)


def reconstruct_texture(resolution, pyramid: mlp.LatentPyramid, mlp_decoder, pe: mlp.TiledPositionalEncoder,
                        device, mip=0, batch_size=65536):
    """Decompress one mip level by sampling all pixel UVs through the pyramid."""
    h = w = resolution
    vs = (torch.arange(h, device=device).float() + 0.5) / h
    us = (torch.arange(w, device=device).float() + 0.5) / w
    grid_v, grid_u = torch.meshgrid(vs, us, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)

    max_mip = int(math.log2(pyramid.base_resolution))
    mip_norm_value = mip / max(max_mip, 1)

    pieces = []
    pyramid.eval()
    mlp_decoder.eval()
    with torch.no_grad():
        for i in range(0, uv.shape[0], batch_size):
            batch_uv = uv[i:i + batch_size]
            feat = pyramid.sample(batch_uv, mip, qat_noise=False, hard_quant=True)
            pe_feat = pe(batch_uv, pyramid.base_resolution)
            mip_norm = torch.full((batch_uv.shape[0], 1), mip_norm_value, device=device)
            pieces.append(mlp_decoder(feat, pe_feat, mip_norm))

    pixels = torch.cat(pieces, dim=0)
    return pixels.T.reshape(-1, h, w)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """[C, H, W] in [0,1] -> PIL RGB image."""
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def visualize_latent(latent_param: torch.Tensor) -> torch.Tensor:
    """First 3 channels of a latent grid normalized to [0,1] for display."""
    if latent_param.dim() == 4:
        tensor = latent_param.data[0]
    else:
        tensor = latent_param.data
    viz = tensor[:3]
    viz = (viz - viz.min()) / (viz.max() - viz.min() + 1e-8)
    return viz


def save_comparison(rows, out_path):
    """
    rows: list of dicts, each with keys 'name', 'original' (PIL), 'latent' (PIL),
    'reconstructed' (PIL), 'diff' (PIL). All PILs must be the same size.
    """
    header = 28
    padding = 6
    cell_w, cell_h = rows[0]['original'].size
    n = len(rows)
    cols = ('original', 'latent', 'reconstructed', 'diff')

    grid_w = len(cols) * cell_w + (len(cols) + 1) * padding
    grid_h = n * (cell_h + header + padding) + padding

    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    for i, row in enumerate(rows):
        y = padding + i * (cell_h + header + padding)

        for j, key in enumerate(cols):
            x = padding + j * (cell_w + padding)
            draw.text((x + 4, y), f"{row['name']} - {key}", fill='black')
            grid.paste(row[key], (x, y + header))

    grid.save(out_path)


def diff_image(orig: torch.Tensor, rec: torch.Tensor, amplify: float = 5.0) -> Image.Image:
    """Absolute per-pixel difference, amplified for visibility."""
    diff = (orig - rec).abs().clamp(0, 1) * amplify
    return tensor_to_pil(diff.clamp(0, 1))


def main(resolution=None, num_iter=250000, is_load=True):
    ASSETS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        'base_color': str(ASSETS_LOAD_DIR / "Default_albedo.jpg"),
        'normal': str(ASSETS_LOAD_DIR / "Default_normal.jpg"),
        'ao': str(ASSETS_LOAD_DIR / "Default_AO.jpg"),
        'metal_roughness': str(ASSETS_LOAD_DIR / "Default_metalRoughness.jpg"),
        'emissive': str(ASSETS_LOAD_DIR / "Default_emissive.jpg"),
    }

    print("Loading textures at native resolution..." if resolution is None
          else f"Loading textures at {resolution}x{resolution}...")
    bundle = load_pbr_texture(paths, resolution=resolution)
    print(f"Texture bundle shape: {tuple(bundle.shape)}")

    resolution = bundle.shape[-1]

    if is_load:
        pyramid, mlp_decoder, pe = load_ntc(ASSETS_EXPORT_DIR / "ntc.pt")
        device = next(mlp_decoder.parameters()).device
    else:
        pyramid, mlp_decoder, pe = train_ntc(bundle, num_iter=num_iter)
        device = next(mlp_decoder.parameters()).device
        print("Saving compressed NTC model...")
        export_ntc(pyramid, mlp_decoder, pe, str(ASSETS_EXPORT_DIR / "ntc.pt"))

        print("Saving runtime layout for C++ ...")
        export_runtime(pyramid, mlp_decoder, pe, ASSETS_EXPORT_DIR / "runtime")

    print("Reconstructing full textures...")
    reconstructed = reconstruct_texture(resolution, pyramid, mlp_decoder, pe, device, mip=0)

    # Channel layout matches the order used in load_pbr_texture
    channel_splits = [
        ('base_color', 0, 3),
        ('normal', 3, 6),
        ('ao', 6, 9),
        ('metal_roughness', 9, 12),
        ('emissive', 12, 15),
    ]

    # Latent visualization uses level 0 (the level that serves the highest mips).
    level0 = pyramid.levels[0]
    hi_pil = tensor_to_pil(visualize_latent(level0.g0))
    lo_pil = tensor_to_pil(visualize_latent(level0.g1))
    hi_pil.save(ASSETS_EXPORT_DIR / "latent_hi.png")
    lo_pil.save(ASSETS_EXPORT_DIR / "latent_lo.png")
    latent_display = hi_pil.resize((resolution, resolution), Image.NEAREST)

    rows = []
    print("Final reconstruction PSNR (vs original, full-res):")
    for name, start, end in channel_splits:
        orig_tensor = bundle[start:end]
        orig_pil = tensor_to_pil(orig_tensor)
        rec_tensor = reconstructed[start:end].cpu()
        rec_pil = tensor_to_pil(rec_tensor)
        diff_pil = diff_image(orig_tensor, rec_tensor)

        mse = F.mse_loss(rec_tensor.clamp(0, 1), orig_tensor).item()
        psnr = float('inf') if mse == 0 else -10.0 * np.log10(mse)
        print(f"  {name:12s}: PSNR {psnr:.2f} dB  (MSE {mse:.6f})")

        rec_pil.save(ASSETS_EXPORT_DIR / f"reconstructed_{name}.png")
        diff_pil.save(ASSETS_EXPORT_DIR / f"diff_{name}.png")

        rows.append({
            'name': f"{name}  PSNR {psnr:.2f} dB",
            'original': orig_pil,
            'latent': latent_display,
            'reconstructed': rec_pil,
            'diff': diff_pil,
        })

    overall_mse = F.mse_loss(reconstructed.cpu().clamp(0, 1), bundle).item()
    overall_psnr = float('inf') if overall_mse == 0 else -10.0 * np.log10(overall_mse)
    print(f"  {'overall':12s}: PSNR {overall_psnr:.2f} dB  (MSE {overall_mse:.6f})")

    save_comparison(rows, str(ASSETS_EXPORT_DIR / "comparison.png"))

    print(f"Done")


if __name__ == '__main__':
    main()
