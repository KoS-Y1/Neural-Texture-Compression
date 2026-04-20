import ShaderIO;
import Utils;

[[vk::push_constant]]
ConstantBuffer<shader_io::GlobalUniforms> pushConstants;

[[vk::binding(0, 0)]]
Sampler2D<float4> skyboxTexture;

struct VSOut
{
    float4 sv_position : SV_Position;
    float3 worldPos;
};

[shader("vertex")]
VSOut vertexMain(float3 inPos: POSITION)
{

    VSOut output;

    const float3 viewSpacePos = mul(inPos, float3x3(pushConstants.view));
    const float4 position = mul(float4(viewSpacePos, 1.0f), pushConstants.proj);

    output.sv_position = position.xyww;
    output.worldPos = inPos;

    return output;
}

[shader("pixel")]
float4 fragmentMain(VSOut input)
{

    float3 worldPos = normalize(input.worldPos);

    const float2 uv = SampleSphericalMap(worldPos);

    float4 skybox = skyboxTexture.SampleLevel(uv, 0);

    return RGBMToLinear(skybox);
}
