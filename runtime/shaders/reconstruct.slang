import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::ReconstructParams> pushConstant;

[[vk::binding(0, 0)]]
Sampler2DArray<float4> latentLoTexture;
[[vk::binding(1, 0)]]
Sampler2DArray<float4> latentHiTexture;
[[vk::binding(2, 0)]]
ByteAddressBuffer mlpBuffer;

[[vk::binding(3, 0)]]
RWTexture2D<float4> outAlbedo;
[[vk::binding(4, 0)]]
RWTexture2D<float4> outNormal;
[[vk::binding(5, 0)]]
RWTexture2D<float4> outAO;
[[vk::binding(6, 0)]]
RWTexture2D<float4> outMattelicRoughness;
[[vk::binding(7, 0)]]
RWTexture2D<float4> outEmissive;

// 4-bit asymmetric dequant: GPU returns code/(N-1) via UNORM8; original real value is
// (sample - 0.5) * (N-1)/N. For N=16 this is 15/16 = 0.9375.
internal static const float kLatentScale = 15.0f / 16.0f;

internal float TriSin(float t) {
    float f = (t + 0.25f) - floor(t + 0.25f);
    return 1.0f - 4.0f * abs(f - 0.5f);
}
internal float TriCos(float t) {
    float f = (t + 0.5f) - floor(t + 0.5f);
    return 1.0f - 4.0f * abs(f - 0.5f);
}


[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= pushConstant.outputWidth || threadId.y >= pushConstant.outputHeight)
    {
        return;
    }

    const float2 uv = (float2(threadId.xy) + 0.5f) / float2(pushConstant.outputWidth, pushConstant.outputHeight);

    CoopVec<float16_t, shader_io::kInDimPadded> inLayer = CoopVec<float16_t, shader_io::kInDimPadded>(float16_t(0.0f));

    // -- G0: concat 4 nearest neighbors (paper Sec 4.3.1). --
    uint hiW, hiH, hiL;
    latentHiTexture.GetDimensions(hiW, hiH, hiL);
    const float2 hiSize  = float2(hiW, hiH);
    const float2 invHi   = 1.0f / hiSize;
    const float2 g0Pos   = uv * hiSize - 0.5f;
    const float2 g0Base  = clamp(floor(g0Pos), float2(0.0f, 0.0f), hiSize - 2.0f);
    const float2 uv00    = (g0Base + 0.5f) * invHi;
    const float2 uv10    = uv00 + float2(invHi.x, 0.0f);
    const float2 uv01    = uv00 + float2(0.0f, invHi.y);
    const float2 uv11    = uv00 + invHi;

    [ForceUnroll]
    for (int li = 0; li < int(shader_io::kLatentDimHi) / 4; ++li) {
        const float4 f00 = (latentHiTexture.SampleLevel(float3(uv00, li), 0) - 0.5f) * kLatentScale;
        const float4 f10 = (latentHiTexture.SampleLevel(float3(uv10, li), 0) - 0.5f) * kLatentScale;
        const float4 f01 = (latentHiTexture.SampleLevel(float3(uv01, li), 0) - 0.5f) * kLatentScale;
        const float4 f11 = (latentHiTexture.SampleLevel(float3(uv11, li), 0) - 0.5f) * kLatentScale;

        const int slot00 = li * 4;
        const int slot10 = int(shader_io::kLatentDimHi) + li * 4;
        const int slot01 = 2 * int(shader_io::kLatentDimHi) + li * 4;
        const int slot11 = 3 * int(shader_io::kLatentDimHi) + li * 4;

        inLayer[slot00 + 0] = float16_t(f00.x);
        inLayer[slot00 + 1] = float16_t(f00.y);
        inLayer[slot00 + 2] = float16_t(f00.z);
        inLayer[slot00 + 3] = float16_t(f00.w);

        inLayer[slot10 + 0] = float16_t(f10.x);
        inLayer[slot10 + 1] = float16_t(f10.y);
        inLayer[slot10 + 2] = float16_t(f10.z);
        inLayer[slot10 + 3] = float16_t(f10.w);

        inLayer[slot01 + 0] = float16_t(f01.x);
        inLayer[slot01 + 1] = float16_t(f01.y);
        inLayer[slot01 + 2] = float16_t(f01.z);
        inLayer[slot01 + 3] = float16_t(f01.w);

        inLayer[slot11 + 0] = float16_t(f11.x);
        inLayer[slot11 + 1] = float16_t(f11.y);
        inLayer[slot11 + 2] = float16_t(f11.z);
        inLayer[slot11 + 3] = float16_t(f11.w);
    }

    const int loBase = 4 * int(shader_io::kLatentDimHi);

    // -- G1: bilinear sample. --
    [ForceUnroll]
    for (int li = 0; li < int(shader_io::kLatentDimLo) / 4; ++li) {
        const float4 lo = (latentLoTexture.SampleLevel(float3(uv, li), 0) - 0.5f) * kLatentScale;
        inLayer[loBase + li * 4 + 0] = float16_t(lo.x);
        inLayer[loBase + li * 4 + 1] = float16_t(lo.y);
        inLayer[loBase + li * 4 + 2] = float16_t(lo.z);
        inLayer[loBase + li * 4 + 3] = float16_t(lo.w);
    }

    // -- Tiled triangular PE. cycles == one tile of 8 texels at base mip. --
    // G0 res is base/4, so peTileFreq = base/8 = hiW/2.
    const float peTileFreq = float(hiW) * 0.5f;
    const float2 cycles = uv * peTileFreq;
    const int peBase = loBase + int(shader_io::kLatentDimLo);
    [ForceUnroll]
    for (int axis = 0; axis < 2; ++axis) {
        const float t = (axis == 0) ? cycles.x : cycles.y;
        [ForceUnroll]
        for (int k = 0; k < int(shader_io::kPeOctaves); ++k) {
            const float phase = t * float(1u << k);
            const int o = peBase + axis * 2 * int(shader_io::kPeOctaves) + k * 2;
            inLayer[o + 0] = float16_t(TriSin(phase));
            inLayer[o + 1] = float16_t(TriCos(phase));
        }
    }

    inLayer[shader_io::kInDim - 1] = float16_t(pushConstant.mipNorm);

    const CoopVec<float16_t, shader_io::kOutDim> outLayer =
        MlpDecoder<shader_io::kInDimPadded, shader_io::kOutDim, shader_io::kHiddenDim>(inLayer, mlpBuffer,
                                                                                       pushConstant.mlpParams);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = float(outLayer[i]);
    }

    outAlbedo[threadId.xy] = float4(output[0], output[1], output[2], 1.0f);
    outNormal[threadId.xy] = float4(output[3], output[4], output[5], 1.0f);
    outAO[threadId.xy] = float4(output[6], output[7], output[8], 1.0f);
    outMattelicRoughness[threadId.xy] = float4(output[9], output[10], output[11], 1.0f);
    outEmissive[threadId.xy] = float4(output[12], output[13], output[14], 1.0f);
}