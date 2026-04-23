implementing Utils;

import ShaderIO;

internal CoopVec<float16_t, OUT> LinearLayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                          const ByteAddressBuffer mlp,
                                                                          const uint weightOffset,
                                                                          const uint biaisOffset)
{
    return coopVecMatMulAddPacked<float16_t, OUT>(
        input, CoopVecComponentType::Float16, IN, mlp, weightOffset, CoopVecComponentType::Float16, mlp, biaisOffset,
        CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false, IN * 2);
}

internal CoopVec<float16_t, OUT> ReLULayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                        const ByteAddressBuffer mlp,
                                                                        const uint weightOffset, const uint biasOffset)
{
    return max(LinearLayer<IN, OUT>(input, mlp, weightOffset, biasOffset), CoopVec<float16_t, OUT>(float16_t(0.0f)));
}

internal CoopVec<float16_t, N> Sigmoid<let N : int>(const CoopVec<float16_t, N> input)
{
    const CoopVec<float16_t, N> c1 = CoopVec<float16_t, N>(float16_t(1.0f));
    return c1 /
           (c1 + exp(-clamp(input, CoopVec<float16_t, N>(float16_t(-10.0f)), CoopVec<float16_t, N>(float16_t(10.0f)))));
}

internal CoopVec<float16_t, OUT> HardGELULayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                            const ByteAddressBuffer mlp,
                                                                            const uint weightOffset,
                                                                            const uint biasOffset)
{
    // HardGELU(x):
    //   x <= -1.5        -> 0
    //   -1.5 < x < 1.5   -> x * (x + 1.5) / 3
    //   x >= 1.5         -> x
    // Branch-free, element-wise identity:
    //   HardGELU(x) = middle(clamp(x, -1.5, 1.5)) + max(x - 1.5, 0)
    const CoopVec<float16_t, OUT> x = LinearLayer<IN, OUT>(input, mlp, weightOffset, biasOffset);
    const CoopVec<float16_t, OUT> cmp = CoopVec<float16_t, OUT>(float16_t(1.5f));
    const CoopVec<float16_t, OUT> three = CoopVec<float16_t, OUT>(float16_t(3.0f));
    const CoopVec<float16_t, OUT> zero = CoopVec<float16_t, OUT>(float16_t(0.0f));

    const CoopVec<float16_t, OUT> xc = clamp(x, -cmp, cmp);
    const CoopVec<float16_t, OUT> middle = (xc * (xc + cmp)) / three;
    const CoopVec<float16_t, OUT> tail = max(x - cmp, zero);
    return middle + tail;
}

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
// Hard coded becasue we know our MLP has 2 hidden layer and the in/out/hidden dims
public CoopVec<float16_t, OUT> MlpDecoder<let IN : int, let OUT : int, let HIDDEN : int>(
    const CoopVec<float16_t, IN> input, const ByteAddressBuffer mlp, const shader_io::MlpParams params)
{
    // Hidden layers
    // CoopVec<float16_t, HIDDEN> h0 = ReLULayer<IN, HIDDEN>(input, mlp, params.w0, params.b0);
    // CoopVec<float16_t, HIDDEN> h1 = ReLULayer<HIDDEN, HIDDEN>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, HIDDEN> h0 = HardGELULayer<IN, HIDDEN>(input, mlp, params.w0, params.b0);
    CoopVec<float16_t, HIDDEN> h1 = HardGELULayer<HIDDEN, HIDDEN>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, OUT> raw = LinearLayer<HIDDEN, OUT>(h1, mlp, params.w2, params.b2);
    // return Sigmoid<OUT>(raw);
    return clamp(raw, CoopVec<float16_t, OUT>(float16_t(0.0f)), CoopVec<float16_t, OUT>(float16_t(1.0f)));
}

public float[shader_io::kOutDim] MLPSample(const float2 uv, const shader_io::ReconstructParams params,
                                           const ByteAddressBuffer mlpBuffer,
                                           const Sampler2DArray<float4> latentLoTexture,
                                           const Sampler2DArray<float4> latentHiTexture)
{

    CoopVec<float16_t, shader_io::kInDimPadded> inLayer = CoopVec<float16_t, shader_io::kInDimPadded>(float16_t(0.0f));

    // -- G0: concat 4 nearest neighbors (paper Sec 4.3.1). --
    uint hiW, hiH, hiL;
    latentHiTexture.GetDimensions(hiW, hiH, hiL);
    const float2 hiSize = float2(hiW, hiH);
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
    for (int li = 0; li < int(shader_io::kLatentDimLo) / 4; ++li)
    {
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
    for (int axis = 0; axis < 2; ++axis)
    {
        const float t = (axis == 0) ? cycles.x : cycles.y;
        [ForceUnroll]
        for (int k = 0; k < int(shader_io::kPeOctaves); ++k)
        {
            const float phase = t * float(1u << k);
            const int o = peBase + axis * 2 * int(shader_io::kPeOctaves) + k * 2;
            inLayer[o + 0] = float16_t(TriSin(phase));
            inLayer[o + 1] = float16_t(TriCos(phase));
        }
    }

    inLayer[shader_io::kInDim - 1] = float16_t(params.mipNorm);

    const CoopVec<float16_t, shader_io::kOutDim> outLayer =
        MlpDecoder<shader_io::kInDimPadded, shader_io::kOutDim, shader_io::kHiddenDim>(inLayer, mlpBuffer,
                                                                                       params.mlpParams);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = float(outLayer[i]);
    }

    return output;
}
