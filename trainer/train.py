import json
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

import latent_texture
import mlp

PROJECT_ROOT = _THIS_DIR.parent
ASSETS_LOAD_DIR = PROJECT_ROOT / "assets/source"
ASSETS_EXPORT_DIR = PROJECT_ROOT / "assets/export"


def train_ntc(texture_bundle: torch.Tensor, num_iter=20000, batch_size=65536, lr_latent=0.01, lr_mlp=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = texture_bundle.shape[0]  # Channels of the texture bundle

    def get_latent_res(texture_res: int, compression_ratio=4, lo_scale=8):
        hi_res = texture_res // compression_ratio
        lo_res = texture_res // lo_scale

        return hi_res, lo_res

    hi_res, lo_res = get_latent_res(texture_bundle.shape[1])
    hi_channels, lo_channels = 4, 4  # Latent grid channels (compression hyperparam)
    num_freq = 8

    # Build mip map chain
    mips = latent_texture.gen_mipmaps(texture_bundle)
    num_mip_levels = len(mips)
    mips = [m.to(device) for m in mips]

    # Initialize modules
    latent_tex = latent_texture.LatentTexture(hi_res, hi_channels, lo_res, lo_channels).to(device)
    pe: mlp.PositionalEncoder = mlp.PositionalEncoder(num_freq).to(device)
    mlp_decoder = mlp.MlpDecoder(
        latent_dim=hi_channels + lo_channels,
        pe_dim=pe.out_dim,
        output_channels=num_channels,
        hidden_dim=64,
        num_hidden=2
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': latent_tex.parameters(), 'lr': lr_latent},
        {'params': mlp_decoder.parameters(), 'lr': lr_mlp},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=0)

    for step in range(num_iter):
        mip_idx = torch.randint(0, num_mip_levels, (1,)).item()
        mip_texture = mips[mip_idx]  # [C, H_m, W_m]
        H_m, W_m = mip_texture.shape[1], mip_texture.shape[2]

        # Radom pixel positon
        px = torch.randint(0, W_m, (batch_size,), device=device)
        py = torch.randint(0, H_m, (batch_size,), device=device)

        # Remape to [0, 1] uv space
        u = (px.float() + 0.5) / W_m
        v = (py.float() + 0.5) / H_m
        uv = torch.stack([u, v], dim=-1)  # [B, 2]

        ground_truth = mip_texture[:, py, px].T  # [B, C]

        # Forward pass
        latent_feature = latent_tex.sample(uv)
        pe_feature = pe(uv)
        mip_norm = torch.full(
            (batch_size, 1),
            mip_idx / max(num_mip_levels - 1, 1),
            device=device,
        )
        pred = mlp_decoder(latent_feature, pe_feature, mip_norm)

        loss = F.mse_loss(pred, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == num_iter - 1:
            psnr = -10 * torch.log10(loss).item()
            print(f"Step {step} / {num_iter} | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")

    return latent_tex, mlp_decoder, pe


def quantize_ste(x, num_bits=8):
    """Quantization with straight-through estimator (used during training)."""
    qmax = (2 ** num_bits) - 1

    x_clamped = torch.clamp(x, 0, 1)
    x_quant = torch.round(x_clamped * qmax) / qmax

    return x_clamped + (x_quant - x_clamped).detach()


def quantize_export(x, num_bits):
    """Hard quantization to integer codes for on-disk storage."""
    qmax = (2 ** num_bits) - 1
    x_clamped = torch.clamp(x, 0, 1)
    codes = (x_clamped * qmax).round()
    dtype = torch.uint8 if num_bits <= 8 else torch.int16
    return codes.to(dtype)


def export_ntc(latent_tex: latent_texture.LatentTexture, mlp_decoder: mlp.MlpDecoder, pe: mlp.PositionalEncoder, path):
    torch.save({
        'latent_hi': quantize_export(latent_tex.latent_hi.data, num_bits=8),
        'latent_hi_bits': 8,
        'latent_lo': quantize_export(latent_tex.latent_lo.data, num_bits=4),
        'latent_lo_bits': 4,
        'mlp_weights': mlp_decoder.state_dict(),
        'pe_num_freq': pe.num_freq,
        'config': {
            'hi_res': latent_tex.latent_hi.shape[-1],
            'lo_res': latent_tex.latent_lo.shape[-1],
            'hi_channels': latent_tex.latent_hi.shape[1],
            'lo_channels': latent_tex.latent_lo.shape[1],
        }
    }, path)


def _quantize_to_unorm8(x: torch.Tensor, num_bits: int) -> np.ndarray:
    """Quantize to num_bits levels, then remap to full [0, 255] uint8 range.

    This preserves the training-time quantization error while letting Vulkan
    sample the texture as R8_UNORM and receive values in [0, 1] directly.
    """
    qmax = (2 ** num_bits) - 1
    x_clamped = torch.clamp(x, 0, 1)
    codes = torch.round(x_clamped * qmax) / qmax
    return (codes * 255.0).round().to(torch.uint8).cpu().numpy()


def _latent_to_interleaved_hwc(latent: torch.Tensor, num_bits: int) -> np.ndarray:
    """[1, C, H, W] latent parameter -> [H, W, C] uint8 buffer, UNORM-ready."""
    tensor = latent.detach()[0]  # [C, H, W]
    quantized = _quantize_to_unorm8(tensor, num_bits)  # [C, H, W]
    return np.ascontiguousarray(np.transpose(quantized, (1, 2, 0)))  # [H, W, C]


def export_runtime(
    latent_tex: latent_texture.LatentTexture,
    mlp_decoder: mlp.MlpDecoder,
    pe: mlp.PositionalEncoder,
    out_dir: Path,
    hi_bits: int = 8,
    lo_bits: int = 4,
):
    """Export latent grids, MLP weights, and a JSON header for the C++ runtime.

    Layout:
      out_dir/
        ntc.json        -- config, layer shapes, byte offsets
        latent_hi.bin   -- uint8, row-major [H, W, C], UNORM-ready
        latent_lo.bin   -- uint8, row-major [H, W, C], UNORM-ready
        mlp.bin         -- float32, concatenated Linear layers in forward order,
                           each layer stored as weight [out, in] row-major then
                           bias [out].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    hi_hwc = _latent_to_interleaved_hwc(latent_tex.latent_hi, hi_bits)
    lo_hwc = _latent_to_interleaved_hwc(latent_tex.latent_lo, lo_bits)
    (out_dir / "latent_hi.bin").write_bytes(hi_hwc.tobytes())
    (out_dir / "latent_lo.bin").write_bytes(lo_hwc.tobytes())

    def _activation_name(m: nn.Module) -> str:
        if isinstance(m, nn.ReLU):
            return "relu"
        if isinstance(m, nn.Sigmoid):
            return "sigmoid"
        return "none"

    linear_layers: list[tuple[nn.Linear, str]] = []
    modules = list(mlp_decoder.net)
    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            activation = _activation_name(modules[i + 1]) if i + 1 < len(modules) else "none"
            linear_layers.append((m, activation))

    mlp_buffers: list[bytes] = []
    mlp_layers_meta = []
    offset = 0
    for linear, activation in linear_layers:
        weight = linear.weight.detach().cpu().contiguous().float().numpy()  # [out, in]
        bias = linear.bias.detach().cpu().contiguous().float().numpy()       # [out]

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

    hi_h, hi_w, hi_c = hi_hwc.shape
    lo_h, lo_w, lo_c = lo_hwc.shape
    input_dim = mlp_layers_meta[0]["in"] if mlp_layers_meta else 0
    output_dim = mlp_layers_meta[-1]["out"] if mlp_layers_meta else 0

    header = {
        "version": 1,
        "latent_hi": {
            "file": "latent_hi.bin",
            "width": hi_w,
            "height": hi_h,
            "channels": hi_c,
            "dtype": "uint8",
            "layout": "hwc_interleaved",
            "sample_format": "unorm",
            "source_bits": hi_bits,
        },
        "latent_lo": {
            "file": "latent_lo.bin",
            "width": lo_w,
            "height": lo_h,
            "channels": lo_c,
            "dtype": "uint8",
            "layout": "hwc_interleaved",
            "sample_format": "unorm",
            "source_bits": lo_bits,
        },
        "positional_encoder": {
            "num_freq": int(pe.num_freq),
            "out_dim": int(pe.out_dim),
        },
        "mlp": {
            "file": "mlp.bin",
            "dtype": "float32",
            "weight_layout": "row_major_out_in",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "total_bytes": offset,
            "layers": mlp_layers_meta,
        },
    }

    with (out_dir / "ntc.json").open("w") as f:
        json.dump(header, f, indent=2)


def decompress_texel(uv, mip_level, latent_tex, mlp_decoder, pe):
    feat = latent_tex.sample(uv)
    pe_feat = pe(uv)
    mip_norm = torch.tensor([[mip_level]], device=uv.device)
    return mlp_decoder(feat, pe_feat, mip_norm)


def reconstruct_texture(resolution, latent_tex, mlp_decoder, pe, device, mip_level=0.0, batch_size=65536):
    """Decompress the full texture by sampling all pixel UVs."""
    h = w = resolution
    vs = (torch.arange(h, device=device).float() + 0.5) / h
    us = (torch.arange(w, device=device).float() + 0.5) / w
    grid_v, grid_u = torch.meshgrid(vs, us, indexing='ij')
    uv = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)  # [H*W, 2]

    pieces = []
    latent_tex.eval()
    mlp_decoder.eval()
    with torch.no_grad():
        for i in range(0, uv.shape[0], batch_size):
            batch_uv = uv[i:i + batch_size]
            feat = latent_tex.sample(batch_uv)
            pe_feat = pe(batch_uv)
            mip_norm = torch.full((batch_uv.shape[0], 1), mip_level, device=device)
            pieces.append(mlp_decoder(feat, pe_feat, mip_norm))

    pixels = torch.cat(pieces, dim=0)  # [H*W, C]
    return pixels.T.reshape(-1, h, w)  # [C, H, W]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """[C, H, W] in [0,1] -> PIL RGB image."""
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def visualize_latent(latent_param: torch.nn.Parameter) -> torch.Tensor:
    """Take the first 3 channels of a latent grid and normalize for display as RGB."""
    tensor = latent_param.data[0]  # [C, H, W]
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


def main(resolution=None, num_iter=10000):
    ASSETS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        'base_color': str(ASSETS_LOAD_DIR / "Default_albedo.jpg"),
        'normal': str(ASSETS_LOAD_DIR / "Default_normal.jpg"),
        'ao': str(ASSETS_LOAD_DIR / "Default_AO.jpg"),
        'emissive': str(ASSETS_LOAD_DIR / "Default_emissive.jpg"),
    }

    print("Loading textures at native resolution..." if resolution is None
          else f"Loading textures at {resolution}x{resolution}...")
    bundle = latent_texture.load_pbr_texture(paths, resolution=resolution)
    print(f"Texture bundle shape: {tuple(bundle.shape)}")

    # Derive output resolution from the loaded bundle (H == W is assumed throughout).
    resolution = bundle.shape[-1]

    latent_tex, mlp_decoder, pe = train_ntc(bundle, num_iter=num_iter)

    device = next(mlp_decoder.parameters()).device

    print("Saving compressed NTC model...")
    export_ntc(latent_tex, mlp_decoder, pe, str(ASSETS_EXPORT_DIR / "ntc.pt"))

    print("Saving runtime layout for C++ ...")
    export_runtime(latent_tex, mlp_decoder, pe, ASSETS_EXPORT_DIR / "runtime")

    print("Reconstructing full textures...")
    reconstructed = reconstruct_texture(resolution, latent_tex, mlp_decoder, pe, device)

    # Channel layout matches the order used in load_pbr_texture
    channel_splits = [
        ('base_color', 0, 3),
        ('normal', 3, 6),
        ('ao', 6, 9),
        ('emissive', 9, 12),
    ]

    # Latent visualization (shared across rows)
    hi_viz = visualize_latent(latent_tex.latent_hi)
    lo_viz = visualize_latent(latent_tex.latent_lo)
    hi_pil = tensor_to_pil(hi_viz)
    lo_pil = tensor_to_pil(lo_viz)
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
