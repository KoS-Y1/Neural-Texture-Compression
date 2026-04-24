#if defined(__cplusplus)
#pragma once

#include <glm/glm.hpp>

#define SLANG_PUBLIC

// GLSL Type
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using float4x4 = glm::mat4;

#elif defined(__SLANG__)
implementing ShaderIO;

#define SLANG_PUBLIC public

#endif

namespace shader_io
{

SLANG_PUBLIC struct GlobalUniforms
{
    SLANG_PUBLIC float4x4 view;
    SLANG_PUBLIC float4x4 proj;
    SLANG_PUBLIC float4x4 viewProj;
    SLANG_PUBLIC float4x4 viewInverse;
};

// Hard-coded NTC layer info matching the trainer (paper-aligned: 4 G0 neighbors, tiled triangular PE).
//   kLatentDimHi    = G0 channels (must be multiple of 4 for RGBA layered storage)
//   kLatentDimLo    = G1 channels (must be multiple of 4)
//   4*kLatentDimHi  = G0 contribution after 4-neighbor concat
//   kPeDim          = 2 axes * (sin, cos) * 3 octaves = 12
//   +1              = normalized LOD
//   kInDimPadded    = next multiple of 8 (CoopVec row-alignment requirement)
SLANG_PUBLIC static const uint32_t kLatentDimHi = 12;
SLANG_PUBLIC static const uint32_t kLatentDimLo = 20;
SLANG_PUBLIC static const uint32_t kPeOctaves = 3;
SLANG_PUBLIC static const uint32_t kPeDim = 2 * 2 * kPeOctaves;
SLANG_PUBLIC static const uint32_t kInDim = 4 * kLatentDimHi + kLatentDimLo + kPeDim + 1;
SLANG_PUBLIC static const uint32_t kInDimPadded = ((kInDim + 7) / 8) * 8;
SLANG_PUBLIC static const uint32_t kHiddenDim = 64;
SLANG_PUBLIC static const uint32_t kOutDim = 15;

SLANG_PUBLIC struct MlpParams
{
    SLANG_PUBLIC uint32_t w0;
    SLANG_PUBLIC uint32_t w1;
    SLANG_PUBLIC uint32_t w2;
    SLANG_PUBLIC uint32_t b0;
    SLANG_PUBLIC uint32_t b1;
    SLANG_PUBLIC uint32_t b2;

    uint32_t _pad0;
    uint32_t _pad1;
};

SLANG_PUBLIC struct ReconstructParams
{
    SLANG_PUBLIC MlpParams mlpParams;
    SLANG_PUBLIC uint32_t outputWidth;
    SLANG_PUBLIC uint32_t outputHeight;
    SLANG_PUBLIC float mipNorm;
};

}
