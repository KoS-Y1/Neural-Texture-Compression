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

// Hard coded layer info
SLANG_PUBLIC static const uint32_t kInDim = 41;
// CoopVec row-major matrices need 16-byte row alignment; pad the input vector to 48 FP16s.
SLANG_PUBLIC static const uint32_t kInDimPadded = 48;
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

    uint32_t paddings[2];
};

SLANG_PUBLIC struct ReconstructParams {
    SLANG_PUBLIC MlpParams mlpParams;
    SLANG_PUBLIC uint32_t outputWidth;
    SLANG_PUBLIC uint32_t outputHeight;
    SLANG_PUBLIC float mipNorm;
};

}
