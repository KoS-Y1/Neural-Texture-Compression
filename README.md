## Output Comparison

![Comparison](/assets/export/comparison.png)
*(From left to right: input, latent, reconstructed, diff)*

Final reconstruction PSNR (vs original, full-res):

|                   | PSNR     | MSE      |
|-------------------|----------|----------|
| Albedo            | 27.06 dB | 0.001969 |
| Normal            | 34.38 dB | 0.000365 |
| AO                | 32.10 dB | 0.000616 |
| MetallicRoughness | 29.55 dB | 0.001109 |
| Emissive          | 35.46 dB | 0.000285 |
| Overall           | 30.92 dB | 0.000809 |

## Reference

- [An Introduction to Neural Shading](https://research.nvidia.com/labs/rtr/publication/duca2025neural/)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/)
- [Neural Rendering in NVIDIA OptiX Using Cooperative Vectors](https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/)
