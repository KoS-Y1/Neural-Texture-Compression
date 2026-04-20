implementing Utils;

internal float3 FresnelSchlick(const float cosTheta, const float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

internal float3 FresnelSchlickRoughness(const float cosTheta, const float3 F0, const float roughness)
{
    return F0 + (max(float3(1.0f - roughness), F0) - F0) * pow(saturate(1.0f - cosTheta), 5.0f);
}

internal float DistributionGGX(const float3 N, const float3 H, const float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    return a2 / (PI * denom * denom);
}

internal float GeometrySchlickGGX(const float NdotV, const float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float denom = NdotV * (1.0f - k) + k;
    return NdotV / denom;
}

internal float GeometrySmith(const float3 N, const float3 V, const float3 L, const float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

public float3 CalculateBRDF(const float3 N, const float3 V, const float3 L, const float3 albedo, const float roughness,
                           const float metallic, const float3 lightColor, const float lightIntensity)
{
    float3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0f);
    float NdotV = max(dot(N, V), 0.0f);

    // Fresnel reflectance at normal incidence
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0f), F0);

    float G = GeometrySmith(N, V, L, roughness);
    float NDF = DistributionGGX(N, H, roughness);

    float3 kS = F;
    float3 kD = (float3(1.0f) - kS) * (1.0f - metallic);

    float3 numerator = NDF * G * F;
    float denominator = 4.0f * NdotV * NdotL + 0.001f; // Prevent division by zero
    float3 specular = numerator / denominator;

    float3 irradiance = lightColor * lightIntensity;

    return (kD * albedo / PI + specular) * irradiance * NdotL;
}
