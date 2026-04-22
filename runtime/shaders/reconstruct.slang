import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::ReconstructParams> pushConstant;

[[vk::binding(0, 0)]]
Sampler2D<float4> latentLoTexture;
[[vk::binding(1, 0)]]
Sampler2D<float4> latentHiTexture;
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

// Hard coded because I'm lazy
internal static const int kLatentDim = 8; // lo_channels + hi_channels
internal static const int kNumFreq = 8;
internal static const int kPeDim = 2 * 2 * kNumFreq;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= pushConstant.outputWidth || threadId.y >= pushConstant.outputHeight)
    {
        return;
    }

    const float2 uv = float2(threadId.xy + 0.5f) / float2(pushConstant.outputWidth, pushConstant.outputHeight);

    CoopVec<float16_t, shader_io::kInDimPadded> inLayer = CoopVec<float16_t, shader_io::kInDimPadded>(float16_t(0.0f));

    const float4 lo = latentLoTexture.SampleLevel(uv, 0);
    const float4 hi = latentHiTexture.SampleLevel(uv, 0);
    inLayer[0] = float16_t(hi.x);
    inLayer[1] = float16_t(hi.y);
    inLayer[2] = float16_t(hi.z);
    inLayer[3] = float16_t(hi.w);
    inLayer[4] = float16_t(lo.x);
    inLayer[5] = float16_t(lo.y);
    inLayer[6] = float16_t(lo.z);
    inLayer[7] = float16_t(lo.w);

    // PE
    [ForceUnroll]
    for (int i = 0; i < kNumFreq; ++i)
    {
        const float f = float(1u << i) * PI;

        inLayer[kLatentDim + i] = float16_t(sin(uv.x * f));
        inLayer[kLatentDim + i + kNumFreq] = float16_t(sin(uv.y * f));
        inLayer[kLatentDim + i + kNumFreq * 2] = float16_t(cos(uv.x * f));
        inLayer[kLatentDim + i + kNumFreq * 3] = float16_t(cos(uv.y * f));
    }

    // Mip (stays at the trained position; padded slots kInDim..kInDimPadded-1 remain zero)
    inLayer[shader_io::kInDim - 1] = float16_t(pushConstant.mipNorm);

    const CoopVec<float16_t, shader_io::kOutDim> outLayer =
        MlpDecoder<shader_io::kInDimPadded, shader_io::kOutDim, shader_io::kHiddenDim>(inLayer, mlpBuffer,
                                                                                       pushConstant.mlpParams);

    // Unpack
    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < shader_io::kOutDim; ++i)
    {
        output[i] = float(outLayer[i]);
    }

    outAlbedo[threadId.xy] = float4(output[0], output[1], output[2], 1.0f);
    outNormal[threadId.xy] = float4(output[3], output[4], output[5], 1.0f);
    outAO[threadId.xy] = float4(output[6], output[7], output[8], 1.0f);
    outMattelicRoughness[threadId.xy] = float4(output[9], output[10], output[11], 1.0f);
    outEmissive[threadId.xy] = float4(output[12], output[13], output[14], 1.0f);
}
