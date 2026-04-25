## Output Comparison

![Comparison](/assets/export/comparison.png)
*(From left to right: input, latent, reconstructed, diff)*

Final reconstruction PSNR (vs original, full-res):

|                   | PSNR     | MSE      |
|-------------------|----------|----------|
| Albedo            | 34.78 dB | 0.000332 |
| Normal            | 39.06 dB | 0.000124 |
| AO                | 39.55 dB | 0.000111 |
| MetallicRoughness | 35.56 dB | 0.000278 |
| Emissive          | 46.95 dB | 0.000020 |
| Overall           | 37.62 dB | 0.000173 |

## Reference

- [An Introduction to Neural Shading](https://research.nvidia.com/labs/rtr/publication/duca2025neural/)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/)
- [Neural Rendering in NVIDIA OptiX Using Cooperative Vectors](https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/)
- [vk_cooperative_matrix_perf](https://github.com/jeffbolznv/vk_cooperative_matrix_perf)
  