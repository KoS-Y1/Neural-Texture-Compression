implementing Utils;

// From https://learnopengl.com/PBR/IBL/Specular-IBL
public float2 SampleSphericalMap(float3 input)
{
    const float2 invAtan = float2(0.1591, 0.3183);

    float2 uv = float2(atan2(input.z, input.x), asin(input.y));
    uv *= invAtan;

    return uv + 0.5;
}

public float4 RGBMToLinear(float4 input)
{
    return float4(input.xyz * input.w * 6.0f, 1.0f);
}

internal float3 SampleIBLReflection(const float3 R, const float roughness, const Sampler2D<float4> specularTex)
{
    // based on https://github.com/oframe/ibl-converter/blob/master/src/shaders/PBRShader.js
    const float MAX_REFLECTION_LOD = 6.0;
    float blend = roughness * MAX_REFLECTION_LOD;
    const float level0 = floor(blend);
    const float level1 = min(MAX_REFLECTION_LOD, level0 + 1.0);
    blend -= level0;

    float2 uvSpec = SampleSphericalMap(R);
    uvSpec.y /= 2.0f;

    float2 uv0 = uvSpec;
    float2 uv1 = uvSpec;

    uv0 /= pow(2.0, level0);
    uv0.y += 1.0 - exp(-LN2 * level0);

    uv1 /= pow(2.0, level1);
    uv1.y += 1.0 - exp(-LN2 * level1);

    float3 specular0 = RGBMToLinear(specularTex.SampleLevel(uv0, 0)).rgb;
    float3 specular1 = RGBMToLinear(specularTex.SampleLevel(uv1, 0)).rgb;
    float3 reflection = lerp(specular0, specular1, blend);

    return reflection;
}

public float3 CalculateIBL(const float3 N, const float NdotV, const float3 R, const float3 F0, const float3 albedo,
                  const float roughness, const float metallic, const float ao, const Sampler2D<float4> irradianceTex,
                  const Sampler2D<float4> specularTex, const Sampler2D<float4> brdfLutTex)
{
    const float3 kS = FresnelSchlickRoughness(NdotV, F0, roughness);
    const float3 kD = (float3(1.0f) - kS) * (1.0f - metallic);

    const float3 irradiance = irradianceTex.SampleLevel(SampleSphericalMap(N), 0).xyz;
    const float3 reflection = SampleIBLReflection(R, roughness, specularTex);
    const float2 brdf = brdfLutTex.SampleLevel(float2(NdotV, roughness), 0).rg;
    const float3 specular = reflection * (kS * brdf.x + brdf.y);

    return (kD * irradiance * albedo + specular) * ao;
}
