import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::GlobalUniforms> pushConstants;

// IBL
[[vk::binding(0, 0)]]
Sampler2D<float4> irradianceTexture;
[[vk::binding(1, 0)]]
Sampler2D<float4> specularTexture;
[[vk::binding(2, 0)]]
Sampler2D<float4> brdfLutTexture;

[[vk::binding(3, 0)]]
Sampler2DArray<float4> latentLoTexture;
[[vk::binding(4, 0)]]
Sampler2DArray<float4> latentHiTexture;
[[vk::binding(5, 0)]]
ByteAddressBuffer mlpBuffer;
[[vk::binding(6, 0)]]
ConstantBuffer<shader_io::ReconstructParams> reconstructParams;

struct VSIn
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 uv : TEXCOORD0;
};

struct VSOut
{
    float4 sv_position : SV_Position;
    float3 worldPos;
    float2 uv;
    float3x3 TBN;
}

[shader("vertex")]
VSOut vertexMain(const VSIn input)
{
    VSOut output;

    const float3x3 normalMatrix = float3x3(kIdentity);

    float3 T, B;
    const float3 N = normalize(mul(input.normal, normalMatrix));

    if (length(input.tangent) > 0.0001f)
    {
        T = normalize(mul(input.tangent, normalMatrix));
        B = cross(N, T);
    }
    else
    {
        float3 up = abs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
        T = normalize(cross(up, N));
        B = cross(N, T);
    }

    output.worldPos = mul(float4(input.pos, 1.0f), kIdentity).xyz;
    output.sv_position = mul(float4(output.worldPos, 1.0f), pushConstants.viewProj);
    output.TBN = float3x3(T, B, N);
    output.uv = input.uv;

    return output;
}

[shader("pixel")]
float4 fragmentMain(VSOut vsStage)
{
    float2 uv = vsStage.uv;
    uv.y = -uv.y;

    const float output[shader_io::kOutDim] =
        MlpSampleCoopVec(uv, reconstructParams, mlpBuffer, latentLoTexture, latentHiTexture);

    const float3 srgbAlbedo = float3(output[0], output[1], output[2]);
    const float3 albedo = pow(abs(srgbAlbedo), 2.2f);
    float3 normal = float3(output[3], output[4], output[5]);
    normal = normal * 2.0f - 1.0f;
    const float ao = output[6];
    const float metallic = output[11];
    const float roughness = output[10];
    const float3 emissive = float3(output[12], output[13], output[14]);

    // Fixed light because I'm lazy
    const float3 lightDir = normalize(float3(0.5f, 1.0f, 0.5f));
    const float3 lightColor = float3(1.0f);
    const float lightIntensity = 3.0f;

    const float3 cameraPos = pushConstants.viewInverse[3].xyz;
    const float3 N = normalize(mul(normal, vsStage.TBN));
    const float3 V = normalize(cameraPos - vsStage.worldPos);
    const float3 L = lightDir;
    const float3 R = reflect(-V, N);
    const float NdotV = max(dot(N, V), 0.0f);
    const float3 F0 = lerp(float3(0.04f), albedo, metallic);

    const float3 pbr = CalculateBRDF(N, V, L, albedo, roughness, metallic, lightColor, lightIntensity);
    const float3 ibl = CalculateIBL(N, NdotV, R, F0, albedo, roughness, metallic, ao, irradianceTexture,
                                    specularTexture, brdfLutTexture);

    return float4(pbr + ibl * 0.5 + emissive, 1.0f);
}
