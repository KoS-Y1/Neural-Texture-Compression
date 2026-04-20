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
}
