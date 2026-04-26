# Neural Rendering

A neural texture compression and rendering experiment: an MLP is trained offline in PyTorch to compress PBR material textures into a compact latent representation, then reconstructed at runtime in a Vulkan + [Slang](https://github.com/shader-slang/slang) renderer using cooperative vector / cooperative matrix acceleration.

## Offline Output Comparison

The model is trained offline in PyTorch. The figure below shows the per-channel output:

![Comparison](/assets/export/comparison.png)
*(From left to right: input, latent, reconstructed, diff)*

Final reconstruction PSNR (vs. original, full-res):

|                   | PSNR     | MSE      |
|-------------------|----------|----------|
| Albedo            | 34.78 dB | 0.000332 |
| Normal            | 39.06 dB | 0.000124 |
| AO                | 39.55 dB | 0.000111 |
| MetallicRoughness | 35.56 dB | 0.000278 |
| Emissive          | 46.95 dB | 0.000020 |
| **Overall**       | **37.62 dB** | **0.000173** |

## Runtime Inference

The trained MLP is evaluated at runtime with Vulkan and [Slang](https://github.com/shader-slang/slang). As shown below, the runtime-reconstructed textures closely match the originals.

![Runtime Comparison](/docs/ntc.png)
*(Left: original textures; right: runtime reconstructed textures)*

Pre-reconstructing the full texture set with a compute shader using **cooperative vector** takes **0.744 ms** with the following PSNR (vs. original, full-res):

|                   | PSNR     |
|-------------------|----------|
| Albedo            | 35.34 dB |
| Normal            | 39.54 dB |
| AO                | 39.78 dB |
| MetallicRoughness | 35.87 dB |
| Emissive          | 47.03 dB |
| **Overall**       | **38.03 dB** |

### Full-screen Inference

Per-frame cost when reconstructing at run-time during shading:

![Full-screen Inference](/docs/fullscreen.png)

I tested a baseline shader without acceleration, then with **cooperative vector** and **cooperative matrix** paths:

|                                                       | Performance (GPU) |
|-------------------------------------------------------|-------------------|
| Traditional Forward PBR (with reconstructed textures) | 0.103 ms          |
| Neural Rendering Forward Pass (Coop Vec)              | 0.402 ms          |
| Neural Rendering Deferred Pass (No Acceleration)      | 5.380 ms          |
| Neural Rendering Deferred Pass (Coop Vec)             | 0.474 ms          |
| Neural Rendering Deferred Pass (Coop Mat)             | 1.901 ms          |

### Filtering

Pre-reconstructed textures look fine, but real-time reconstruction shows visible *blocky* artifacts when zoomed in:

![Blocky](/docs/blocky.png)

To reduce the artifacts, I applied bilinear filtering during sampling. Results below (the cooperative matrix deferred path is omitted because it is too slow to be useful here):

![Bilinear](/docs/bilinear.png)

|                                                  | Performance (GPU) |
|--------------------------------------------------|-------------------|
| Traditional Forward PBR (with original textures) | 0.428 ms          |
| Neural Rendering Forward Pass (Coop Vec)         | 1.517 ms          |
| Neural Rendering Deferred Pass (No Acceleration) | 26.868 ms         |
| Neural Rendering Deferred Pass (Coop Vec)        | 1.288 ms          |

## References

- [An Introduction to Neural Shading](https://research.nvidia.com/labs/rtr/publication/duca2025neural/)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/)
- [Neural Rendering in NVIDIA OptiX Using Cooperative Vectors](https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/)
- [vk_cooperative_matrix_perf](https://github.com/jeffbolznv/vk_cooperative_matrix_perf)