import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::GlobalUniforms> pushConstants;

[[vk::binding(0, 0)]]
Sampler2D<float4> albedoTexture;
[[vk::binding(1, 0)]]
Sampler2D<float4> aoTexture;
[[vk::binding(2, 0)]]
Sampler2D<float4> emissiveTexture;
[[vk::binding(3, 0)]]
Sampler2D<float4> metallicRoughnessTexture;
[[vk::binding(4, 0)]]
Sampler2D<float4> normalTexture;
[[vk::binding(5, 0)]]
Sampler2D<float4> irradianceTexture;
[[vk::binding(6, 0)]]
Sampler2D<float4> specularTexture;
[[vk::binding(7, 0)]]
Sampler2D<float4> brdfLutTexture;

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

    static const float4x4 identity = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    const float3x3 normalMatrix = float3x3(identity);

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

    output.worldPos = mul(float4(input.pos, 1.0f), identity).xyz;
    output.sv_position = mul(float4(output.worldPos, 1.0f), pushConstants.viewProj);
    output.TBN = float3x3(T, B, N);
    output.uv = input.uv;

    return output;
}

[shader("pixel")]
float4 fragmentMain(VSOut vsStage)
{
    const float3 albedo = albedoTexture.Sample(vsStage.uv).rgb;
    const float3 mr = metallicRoughnessTexture.Sample(vsStage.uv).rgb;
    const float3 emissive = emissiveTexture.Sample(vsStage.uv).rgb;
    const float roughness = mr.g;
    const float metallic = mr.b;
    const float ao = aoTexture.Sample(vsStage.uv).r;

    const float3 normalTS = normalTexture.Sample(vsStage.uv).xyz * 2.0f - 1.0f;

    // Fixed light because I'm lazy
    const float3 lightDir = normalize(float3(0.5f, 1.0f, 0.5f));
    const float3 lightColor = float3(1.0f);
    const float lightIntensity = 3.0f;

    const float3 cameraPos = pushConstants.viewInverse[3].xyz;
    const float3 N = normalize(mul(normalTS, vsStage.TBN));
    const float3 V = normalize(cameraPos - vsStage.worldPos);
    const float3 L = lightDir;
    const float3 R = reflect(-V, N);
    const float NdotV = max(dot(N, V), 0.0f);
    const float3 F0 = lerp(float3(0.04f), albedo, metallic);

    const float3 pbr = CalculateBRDF(N, V, L, albedo, roughness, metallic, lightColor, lightIntensity);
    const float3 ibl = CalculateIBL(N, NdotV, R, F0, albedo, roughness, metallic, ao, irradianceTexture,
                                    specularTexture, brdfLutTexture);

    return float4(pbr + ibl + emissive, 1.0f);
}
