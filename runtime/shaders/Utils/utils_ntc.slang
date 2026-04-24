implementing Utils;

import ShaderIO;

// 4-bit asymmetric dequant: GPU returns code/(N-1) via UNORM8; original real value is
// (sample - 0.5) * (N-1)/N. For N=16 this is 15/16 = 0.9375.
internal static const float kLatentScale = 15.0f / 16.0f;

internal float TriSin(float t)
{
    float f = (t + 0.25f) - floor(t + 0.25f);
    return 1.0f - 4.0f * abs(f - 0.5f);
}
internal float TriCos(float t)
{
    float f = (t + 0.5f) - floor(t + 0.5f);
    return 1.0f - 4.0f * abs(f - 0.5f);
}

// ---------------------------------------------------------------------------
// Input feature construction: shared between standard and coopvec paths.
// ---------------------------------------------------------------------------
internal float16_t[shader_io::kInDimPadded] BuildInputFeatures(
    const float2 uv, const shader_io::ReconstructParams params,
    const Sampler2DArray<float4> latentLoTexture, const Sampler2DArray<float4> latentHiTexture)
{
    float16_t feats[shader_io::kInDimPadded];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kInDimPadded); ++i)
    {
        feats[i] = float16_t(0.0f);
    }

    // -- G0: concat 4 nearest neighbors (paper Sec 4.3.1). --
    // G0 resolution is always output/4.
    const float2 hiSize = float2(params.outputWidth, params.outputHeight) * 0.25f;
    const float2 invHi = 1.0f / hiSize;
    const float2 g0Pos = uv * hiSize - 0.5f;
    const float2 g0Base = clamp(floor(g0Pos), float2(0.0f, 0.0f), hiSize - 2.0f);
    const float2 uv00 = (g0Base + 0.5f) * invHi;
    const float2 uv10 = uv00 + float2(invHi.x, 0.0f);
    const float2 uv01 = uv00 + float2(0.0f, invHi.y);
    const float2 uv11 = uv00 + invHi;

    [ForceUnroll]
    for (int li = 0; li < int(shader_io::kLatentDimHi) / 4; ++li)
    {
        const float4 f00 = (latentHiTexture.SampleLevel(float3(uv00, li), 0) - 0.5f) * kLatentScale;
        const float4 f10 = (latentHiTexture.SampleLevel(float3(uv10, li), 0) - 0.5f) * kLatentScale;
        const float4 f01 = (latentHiTexture.SampleLevel(float3(uv01, li), 0) - 0.5f) * kLatentScale;
        const float4 f11 = (latentHiTexture.SampleLevel(float3(uv11, li), 0) - 0.5f) * kLatentScale;

        const int slot00 = li * 4;
        const int slot10 = int(shader_io::kLatentDimHi) + li * 4;
        const int slot01 = 2 * int(shader_io::kLatentDimHi) + li * 4;
        const int slot11 = 3 * int(shader_io::kLatentDimHi) + li * 4;

        feats[slot00 + 0] = float16_t(f00.x);
        feats[slot00 + 1] = float16_t(f00.y);
        feats[slot00 + 2] = float16_t(f00.z);
        feats[slot00 + 3] = float16_t(f00.w);

        feats[slot10 + 0] = float16_t(f10.x);
        feats[slot10 + 1] = float16_t(f10.y);
        feats[slot10 + 2] = float16_t(f10.z);
        feats[slot10 + 3] = float16_t(f10.w);

        feats[slot01 + 0] = float16_t(f01.x);
        feats[slot01 + 1] = float16_t(f01.y);
        feats[slot01 + 2] = float16_t(f01.z);
        feats[slot01 + 3] = float16_t(f01.w);

        feats[slot11 + 0] = float16_t(f11.x);
        feats[slot11 + 1] = float16_t(f11.y);
        feats[slot11 + 2] = float16_t(f11.z);
        feats[slot11 + 3] = float16_t(f11.w);
    }

    const int loBase = 4 * int(shader_io::kLatentDimHi);

    // -- G1: bilinear sample. --
    [ForceUnroll]
    for (int li = 0; li < int(shader_io::kLatentDimLo) / 4; ++li)
    {
        const float4 lo = (latentLoTexture.SampleLevel(float3(uv, li), 0) - 0.5f) * kLatentScale;
        feats[loBase + li * 4 + 0] = float16_t(lo.x);
        feats[loBase + li * 4 + 1] = float16_t(lo.y);
        feats[loBase + li * 4 + 2] = float16_t(lo.z);
        feats[loBase + li * 4 + 3] = float16_t(lo.w);
    }

    // -- Tiled triangular PE. peTileFreq = output_res / 8. --
    const float peTileFreq = float(params.outputWidth) * 0.125f;
    const float2 cycles = uv * peTileFreq;
    const int peBase = loBase + int(shader_io::kLatentDimLo);
    [ForceUnroll]
    for (int axis = 0; axis < 2; ++axis)
    {
        const float t = (axis == 0) ? cycles.x : cycles.y;
        [ForceUnroll]
        for (int k = 0; k < int(shader_io::kPeOctaves); ++k)
        {
            const float phase = t * float(1u << k);
            const int o = peBase + axis * 2 * int(shader_io::kPeOctaves) + k * 2;
            feats[o + 0] = float16_t(TriSin(phase));
            feats[o + 1] = float16_t(TriCos(phase));
        }
    }

    feats[shader_io::kInDim - 1] = float16_t(params.mipNorm);
    return feats;
}

// ---------------------------------------------------------------------------
// Standard (scalar) path. Weights are read directly from a ByteAddressBuffer
// (StorageBuffer storage class, covered by storageBuffer16BitAccess). We
// intentionally avoid groupshared float16 because 16-bit Workgroup access
// isn't covered by any core Vulkan feature we have enabled, and NV drivers
// hit undefined/faulting behaviour for heavy scalar fp16 reads from
// Workgroup memory.
// ---------------------------------------------------------------------------
[ForceInline]
internal float16_t[OUT] LinearLayerStandard<let IN : int, let OUT : int>(
    const float16_t[IN] input, const ByteAddressBuffer mlp,
    const uint weightOffset, const uint biasOffset)
{
    float16_t output[OUT];
    for (int j = 0; j < OUT; ++j)
    {
        float16_t acc = mlp.Load<float16_t>(biasOffset + uint(j) * 2);
        const uint rowBytes = weightOffset + uint(j) * uint(IN) * 2;
        for (int i = 0; i < IN; ++i)
        {
            acc += input[i] * mlp.Load<float16_t>(rowBytes + uint(i) * 2);
        }
        output[j] = acc;
    }
    return output;
}

[ForceInline]
internal float16_t[OUT] HardGELULayerStandard<let IN : int, let OUT : int>(
    const float16_t[IN] input, const ByteAddressBuffer mlp,
    const uint weightOffset, const uint biasOffset)
{
    float16_t x[OUT] = LinearLayerStandard<IN, OUT>(input, mlp, weightOffset, biasOffset);

    const float16_t cmp = float16_t(1.5f);
    const float16_t three = float16_t(3.0f);
    const float16_t zero = float16_t(0.0f);

    float16_t output[OUT];
    for (int i = 0; i < OUT; ++i)
    {
        const float16_t xc = clamp(x[i], -cmp, cmp);
        const float16_t middle = (xc * (xc + cmp)) / three;
        const float16_t tail = max(x[i] - cmp, zero);
        output[i] = float16_t(middle + tail);
    }
    return output;
}

[ForceInline]
internal float16_t[shader_io::kOutDim] MlpDecoderStandard(
    const float16_t[shader_io::kInDimPadded] input,
    const ByteAddressBuffer mlp, const shader_io::MlpParams params)
{
    const float16_t h0[shader_io::kHiddenDim] =
        HardGELULayerStandard<shader_io::kInDimPadded, shader_io::kHiddenDim>(input, mlp, params.w0, params.b0);
    const float16_t h1[shader_io::kHiddenDim] =
        HardGELULayerStandard<shader_io::kHiddenDim, shader_io::kHiddenDim>(h0, mlp, params.w1, params.b1);

    float16_t raw[shader_io::kOutDim] =
        LinearLayerStandard<shader_io::kHiddenDim, shader_io::kOutDim>(h1, mlp, params.w2, params.b2);

    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        raw[i] = clamp(raw[i], float16_t(0.0f), float16_t(1.0f));
    }
    return raw;
}

[ForceInline]
public float[shader_io::kOutDim] MlpSampleStandard(
    const float2 uv, const shader_io::ReconstructParams params, const ByteAddressBuffer mlpBuffer,
    const Sampler2DArray<float4> latentLoTexture, const Sampler2DArray<float4> latentHiTexture)
{
    const float16_t inLayer[shader_io::kInDimPadded] =
        BuildInputFeatures(uv, params, latentLoTexture, latentHiTexture);
    const float16_t outLayer[shader_io::kOutDim] = MlpDecoderStandard(inLayer, mlpBuffer, params.mlpParams);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = float(outLayer[i]);
    }
    return output;
}

[ForceInline]
public float[shader_io::kOutDim] MlpSampleBilinearStandard(
    const float2 uv, const shader_io::ReconstructParams params, const ByteAddressBuffer mlpBuffer,
    const Sampler2DArray<float4> latentLoTexture, const Sampler2DArray<float4> latentHiTexture)
{
    const float2 gridSize = float2(params.outputWidth, params.outputHeight);
    const float2 invGrid  = 1.0f / gridSize;
    const float2 p        = uv * gridSize - 0.5f;
    const float2 pf       = floor(p);
    const float2 fr       = p - pf;

    const float2 uv00 = (pf + float2(0.5f, 0.5f)) * invGrid;
    const float2 uv10 = (pf + float2(1.5f, 0.5f)) * invGrid;
    const float2 uv01 = (pf + float2(0.5f, 1.5f)) * invGrid;
    const float2 uv11 = (pf + float2(1.5f, 1.5f)) * invGrid;

    const float w00 = (1.0f - fr.x) * (1.0f - fr.y);
    const float w10 = fr.x * (1.0f - fr.y);
    const float w01 = (1.0f - fr.x) * fr.y;
    const float w11 = fr.x * fr.y;

    const float s00[shader_io::kOutDim] = MlpSampleStandard(uv00, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s10[shader_io::kOutDim] = MlpSampleStandard(uv10, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s01[shader_io::kOutDim] = MlpSampleStandard(uv01, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s11[shader_io::kOutDim] = MlpSampleStandard(uv11, params, mlpBuffer, latentLoTexture, latentHiTexture);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = w00 * s00[i] + w10 * s10[i] + w01 * s01[i] + w11 * s11[i];
    }
    return output;
}

// ---------------------------------------------------------------------------
// CoopVec path. Weights come straight from a ByteAddressBuffer via
// coopVecMatMulAddPacked.
// ---------------------------------------------------------------------------
[ForceInline]
internal CoopVec<float16_t, OUT> LinearLayerCoopVec<let IN : int, let OUT : int>(
    const CoopVec<float16_t, IN> input, const ByteAddressBuffer mlp,
    const uint weightOffset, const uint biasOffset)
{
    return coopVecMatMulAddPacked<float16_t, OUT>(
        input, CoopVecComponentType::Float16, IN, mlp, weightOffset, CoopVecComponentType::Float16, mlp, biasOffset,
        CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false, IN * 2);
}

[ForceInline]
internal CoopVec<float16_t, OUT> HardGELULayerCoopVec<let IN : int, let OUT : int>(
    const CoopVec<float16_t, IN> input, const ByteAddressBuffer mlp,
    const uint weightOffset, const uint biasOffset)
{
    // HardGELU(x):
    //   x <= -1.5        -> 0
    //   -1.5 < x < 1.5   -> x * (x + 1.5) / 3
    //   x >= 1.5         -> x
    // Branch-free form: middle(clamp(x, -1.5, 1.5)) + max(x - 1.5, 0)
    const CoopVec<float16_t, OUT> x = LinearLayerCoopVec<IN, OUT>(input, mlp, weightOffset, biasOffset);

    const CoopVec<float16_t, OUT> cmp = CoopVec<float16_t, OUT>(float16_t(1.5f));
    const CoopVec<float16_t, OUT> three = CoopVec<float16_t, OUT>(float16_t(3.0f));
    const CoopVec<float16_t, OUT> zero = CoopVec<float16_t, OUT>(float16_t(0.0f));

    const CoopVec<float16_t, OUT> xc = clamp(x, -cmp, cmp);
    const CoopVec<float16_t, OUT> middle = (xc * (xc + cmp)) / three;
    const CoopVec<float16_t, OUT> tail = max(x - cmp, zero);
    return middle + tail;
}

[ForceInline]
internal CoopVec<float16_t, shader_io::kOutDim> MlpDecoderCoopVec(
    const CoopVec<float16_t, shader_io::kInDimPadded> input,
    const ByteAddressBuffer mlp, const shader_io::MlpParams params)
{
    CoopVec<float16_t, shader_io::kHiddenDim> h0 =
        HardGELULayerCoopVec<shader_io::kInDimPadded, shader_io::kHiddenDim>(input, mlp, params.w0, params.b0);
    CoopVec<float16_t, shader_io::kHiddenDim> h1 =
        HardGELULayerCoopVec<shader_io::kHiddenDim, shader_io::kHiddenDim>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, shader_io::kOutDim> raw =
        LinearLayerCoopVec<shader_io::kHiddenDim, shader_io::kOutDim>(h1, mlp, params.w2, params.b2);

    return clamp(raw, CoopVec<float16_t, shader_io::kOutDim>(float16_t(0.0f)),
                 CoopVec<float16_t, shader_io::kOutDim>(float16_t(1.0f)));
}

[ForceInline]
public float[shader_io::kOutDim] MlpSampleCoopVec(
    const float2 uv, const shader_io::ReconstructParams params, const ByteAddressBuffer mlpBuffer,
    const Sampler2DArray<float4> latentLoTexture, const Sampler2DArray<float4> latentHiTexture)
{
    const float16_t feats[shader_io::kInDimPadded] =
        BuildInputFeatures(uv, params, latentLoTexture, latentHiTexture);

    CoopVec<float16_t, shader_io::kInDimPadded> inLayer = CoopVec<float16_t, shader_io::kInDimPadded>(float16_t(0.0f));
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kInDimPadded); ++i)
    {
        inLayer[i] = feats[i];
    }

    const CoopVec<float16_t, shader_io::kOutDim> outLayer =
        MlpDecoderCoopVec(inLayer, mlpBuffer, params.mlpParams);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = float(outLayer[i]);
    }
    return output;
}

[ForceInline]
public float[shader_io::kOutDim] MlpSampleBilinearCoopVec(
    const float2 uv, const shader_io::ReconstructParams params, const ByteAddressBuffer mlpBuffer,
    const Sampler2DArray<float4> latentLoTexture, const Sampler2DArray<float4> latentHiTexture)
{
    const float2 gridSize = float2(params.outputWidth, params.outputHeight);
    const float2 invGrid  = 1.0f / gridSize;
    const float2 p        = uv * gridSize - 0.5f;
    const float2 pf       = floor(p);
    const float2 fr       = p - pf;

    const float2 uv00 = (pf + float2(0.5f, 0.5f)) * invGrid;
    const float2 uv10 = (pf + float2(1.5f, 0.5f)) * invGrid;
    const float2 uv01 = (pf + float2(0.5f, 1.5f)) * invGrid;
    const float2 uv11 = (pf + float2(1.5f, 1.5f)) * invGrid;

    const float w00 = (1.0f - fr.x) * (1.0f - fr.y);
    const float w10 = fr.x * (1.0f - fr.y);
    const float w01 = (1.0f - fr.x) * fr.y;
    const float w11 = fr.x * fr.y;

    const float s00[shader_io::kOutDim] = MlpSampleCoopVec(uv00, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s10[shader_io::kOutDim] = MlpSampleCoopVec(uv10, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s01[shader_io::kOutDim] = MlpSampleCoopVec(uv01, params, mlpBuffer, latentLoTexture, latentHiTexture);
    const float s11[shader_io::kOutDim] = MlpSampleCoopVec(uv11, params, mlpBuffer, latentLoTexture, latentHiTexture);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = w00 * s00[i] + w10 * s10[i] + w01 * s01[i] + w11 * s11[i];
    }
    return output;
}