import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::GlobalUniforms> pushConstants;

[[vk::binding(0, 0)]]
Sampler2DArray<float4> latentLoTexture;
[[vk::binding(1, 0)]]
Sampler2DArray<float4> latentHiTexture;
[[vk::binding(2, 0)]]
ByteAddressBuffer mlpBuffer;
[[vk::binding(3, 0)]]
ConstantBuffer<shader_io::ReconstructParams> reconstructParams;

// Gbuffer
[[vk::binding(4, 0)]]
Sampler2D<float4> gbuffer0; // world pos + mesh hit
[[vk::binding(5, 0)]]
Sampler2D<float4> gbuffer1; // TB
[[vk::binding(6, 0)]]
Sampler2D<float4> gbuffer2; // N
[[vk::binding(7, 0)]]
Sampler2D<float4> gbuffer3; // mesh UV

// IBL
[[vk::binding(8, 0)]]
Sampler2D<float4> irradianceTexture;
[[vk::binding(9, 0)]]
Sampler2D<float4> specularTexture;
[[vk::binding(10, 0)]]
Sampler2D<float4> brdfLutTexture;

[[vk::binding(11, 0)]]
RWTexture2D<float4> outputImage;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID, uint groupIndex: SV_GroupIndex)
{

    uint screenWidth, screenHeight;
    outputImage.GetDimensions(screenWidth, screenHeight);
    const bool validPixel = (threadId.x < screenWidth) && (threadId.y < screenHeight);
    const float2 screenUV = validPixel ? (float2(threadId.xy) + 0.5f) / float2(float(screenWidth), float(screenHeight))
                                       : float2(0.0f, 0.0f);

    const float4 worldPosHit = gbuffer0.SampleLevel(screenUV, 0);
    const float2 meshUV = gbuffer3.SampleLevel(screenUV, 0).xy;
    const float output[shader_io::kOutDim] =
        MlpSampleBilinearStandard(meshUV, reconstructParams, mlpBuffer, latentLoTexture, latentHiTexture);

    if (!validPixel)
    {
        return;
    }

    if (worldPosHit.w < 0.5f)
    {
        outputImage[threadId.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    const float3 srgbAlbedo = float3(output[0], output[1], output[2]);
    const float3 albedo = pow(abs(srgbAlbedo), 2.2f);
    float3 normal = float3(output[3], output[4], output[5]);
    normal = normal * 2.0f - 1.0f;
    const float ao = output[6];
    const float metallic = output[11];
    const float roughness = output[10];
    const float3 emissive = float3(output[12], output[13], output[14]);

    const float3 worldPos = worldPosHit.xyz;
    const float3 T = gbuffer1.SampleLevel(screenUV, 0).xyz;
    const float3 inN = gbuffer2.SampleLevel(screenUV, 0).xyz;
    const float3 B = cross(inN, T);
    const float3x3 TBN = float3x3(T, B, inN);

    // Fixed light because I'm lazy
    const float3 lightDir = normalize(float3(0.5f, 1.0f, 0.5f));
    const float3 lightColor = float3(1.0f);
    const float lightIntensity = 3.0f;

    const float3 cameraPos = pushConstants.viewInverse[3].xyz;
    const float3 N = normalize(mul(normal, TBN));
    const float3 V = normalize(cameraPos - worldPos);
    const float3 L = lightDir;
    const float3 R = reflect(-V, N);
    const float NdotV = max(dot(N, V), 0.0f);
    const float3 F0 = lerp(float3(0.04f), albedo, metallic);

    const float3 pbr = CalculateBRDF(N, V, L, albedo, roughness, metallic, lightColor, lightIntensity);
    const float3 ibl = CalculateIBL(N, NdotV, R, F0, albedo, roughness, metallic, ao, irradianceTexture,
                                    specularTexture, brdfLutTexture);

    outputImage[threadId.xy] = float4(pbr + ibl * 0.5f + emissive, 1.0f);
}
