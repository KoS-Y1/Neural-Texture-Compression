import ShaderIO;
import Utils;


[[vk::push_constant]]
ConstantBuffer<shader_io::GlobalUniforms> pushConstants;

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
    float  meshHit;
    float3 N;
    float4 TB;
}

struct FSOut
{
    float4 worldPosMeshHit : SV_Target0;
    float4 TB              : SV_Target1;
    float4 N               : SV_Target2;
    float4 uv              : SV_Target3;
}

[shader("vertex")]
VSOut vertexMain(const VSIn input)
{
    VSOut output;

    const float3x3 normalMatrix = float3x3(kIdentity);

    float3 T;
    const float3 N = normalize(mul(input.normal, normalMatrix));

    if (length(input.tangent) > 0.0001f)
    {
        T = normalize(mul(input.tangent, normalMatrix));
    }
    else
    {
        float3 up = abs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
        T = normalize(cross(up, N));
    }
    float bitangentSign = 1.0f; // handedness unknown without float4 tangent input; always positive

    output.worldPos = mul(float4(input.pos, 1.0f), kIdentity).xyz;
    output.sv_position = mul(float4(output.worldPos, 1.0f), pushConstants.viewProj);
    output.N = N;
    output.TB = float4(T, bitangentSign);
    output.uv = input.uv;
    output.meshHit = 1.0f;

    return output;
}

[shader("pixel")]
FSOut fragmentMain(const VSOut input)
{
    FSOut output;

    output.worldPosMeshHit = float4(input.worldPos, input.meshHit);
    output.TB = input.TB;
    output.N = float4(input.N, 0.0f);
    // Flip uv to match the convention used by forward_neural (uv.y = -uv.y).
    output.uv = float4(input.uv.x, -input.uv.y, 0.0f, 0.0f);

    return output;
}