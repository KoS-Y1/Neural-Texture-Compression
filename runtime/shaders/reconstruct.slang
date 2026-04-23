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

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= pushConstant.outputWidth || threadId.y >= pushConstant.outputHeight)
    {
        return;
    }

    const float2 uv = (float2(threadId.xy) + 0.5f) / float2(pushConstant.outputWidth, pushConstant.outputHeight);

    const float output[shader_io::kOutDim] = MLPSample(uv, pushConstant, mlpBuffer, latentLoTexture, latentHiTexture);

    outAlbedo[threadId.xy] = float4(output[0], output[1], output[2], 1.0f);
    outNormal[threadId.xy] = float4(output[3], output[4], output[5], 1.0f);
    outAO[threadId.xy] = float4(output[6], output[7], output[8], 1.0f);
    outMattelicRoughness[threadId.xy] = float4(output[9], output[10], output[11], 1.0f);
    outEmissive[threadId.xy] = float4(output[12], output[13], output[14], 1.0f);
}
