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

static const uint kNumRows = 32;
static const uint kMt = 16;
static const uint kKt = 16;
static const uint kNt = 16;


static const uint kCoopStride = ((shader_io::kInDimPadded + kKt - 1u) / kKt) * kKt;

groupshared float16_t sharedBuffA[kNumRows * kCoopStride];
groupshared float16_t sharedBuffB[kNumRows * kCoopStride];

internal float16_t HardGELU(float16_t x)
{
    const float16_t cmp = float16_t(1.5f);
    const float16_t three = float16_t(3.0f);
    const float16_t zero = float16_t(0.0f);
    const float16_t xc = clamp(x, -cmp, cmp);
    const float16_t middle = (xc * (xc + cmp)) / three;
    const float16_t tail = max(x - cmp, zero);
    return middle + tail;
}

// Bias add + HardGELU, applied in-place to one row of `buf` (the row owned by this thread).
[ForceInline]
internal void Activation<let N : int>(
    __ref groupshared float16_t[kNumRows * kCoopStride] buf,
    const uint row, const ByteAddressBuffer mlp, const uint biasOffset)
{
    [ForceUnroll]
    for (int j = 0; j < N; ++j)
    {
        const float16_t b = mlp.Load<float16_t>(biasOffset + uint(j) * 2u);
        buf[row * kCoopStride + uint(j)] =
            HardGELU(buf[row * kCoopStride + uint(j)] + b);
    }
}

// Cooperative tiled matmul: output[M x N] = input[M x K] * W[K x N]
[ForceInline]
internal void LinearLayer<let M : int, let K : int, let N : int>(
    __constref groupshared float16_t[kNumRows * kCoopStride] input,
    const ByteAddressBuffer mlp, const uint weightOffset,
    __ref groupshared float16_t[kNumRows * kCoopStride] output)
{
    [ForceUnroll]
    for (uint m = 0u; m < M; m += kMt)
    {
        [ForceUnroll]
        for (uint n = 0u; n < N; n += kNt)
        {
            var matC = linalg::CoopMat<float16_t, MemoryScope::Subgroup, kMt, kNt,
                                       linalg::CoopMatMatrixUse::MatrixAccumulator>(float16_t(0.0f));

            [ForceUnroll]
            for (uint k = 0u; k < K; k += kKt)
            {
                let matA = linalg::CoopMat<float16_t, MemoryScope::Subgroup, kMt, kKt,
                                           linalg::CoopMatMatrixUse::MatrixA>
                    .Load<linalg::CoopMatMatrixLayout::RowMajor>(
                        input, m * kCoopStride + k, kCoopStride);

                // Weights stored row-major as [OUT x IN]. ColumnMajor load with leading
                // dim K transposes to the [IN x OUT] view that MatrixB expects.
                let matB = linalg::CoopMat<float16_t, MemoryScope::Subgroup, kKt, kNt,
                                           linalg::CoopMatMatrixUse::MatrixB>
                    .Load<linalg::CoopMatMatrixLayout::ColumnMajor>(
                        mlp, weightOffset / 2u + n * K + k, K);

                matC = linalg::coopMatMulAdd<float16_t, false>(matA, matB, matC);
            }

            matC.Store<linalg::CoopMatMatrixLayout::RowMajor>(
                output, m * kCoopStride + n, kCoopStride);
        }
    }
}

// One forward pass of the MLP for the meshUV evaluated by this thread.
// All threads must call this together (uniform control flow) because of the cooperative ops.
[ForceInline]
internal float16_t[shader_io::kOutDim] MlpSample(const float2 meshUV, const uint row)
{
    const float16_t feats[shader_io::kInDimPadded] =
        BuildInputFeatures(meshUV, reconstructParams, latentLoTexture, latentHiTexture);

    // Stage features into row's slot, zero-padding cols [kInDimPadded, kCoopStride).
    [ForceUnroll]
    for (uint j = 0u; j < kCoopStride; ++j)
    {
        sharedBuffA[row * kCoopStride + j] =
            (j < shader_io::kInDimPadded) ? feats[j] : float16_t(0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    // Layer 0: sharedBuffA [32 x kInDimPadded] * W0 [kInDimPadded x kHiddenDim] -> sharedBuffB.
    LinearLayer<kNumRows, shader_io::kInDimPadded, shader_io::kHiddenDim>(
        sharedBuffA, mlpBuffer, reconstructParams.mlpParams.w0, sharedBuffB);
    GroupMemoryBarrierWithGroupSync();

    Activation<shader_io::kHiddenDim>(sharedBuffB, row, mlpBuffer, reconstructParams.mlpParams.b0);
    GroupMemoryBarrierWithGroupSync();

    // Layer 1: sharedBuffB [32 x kHiddenDim] * W1 [kHiddenDim x kHiddenDim] -> sharedBuffA.
    LinearLayer<kNumRows, shader_io::kHiddenDim, shader_io::kHiddenDim>(
        sharedBuffB, mlpBuffer, reconstructParams.mlpParams.w1, sharedBuffA);
    GroupMemoryBarrierWithGroupSync();

    Activation<shader_io::kHiddenDim>(sharedBuffA, row, mlpBuffer, reconstructParams.mlpParams.b1);
    GroupMemoryBarrierWithGroupSync();

    // Layer 2: scalar per-thread (kOutDim=15 isn't a coop-mat shape).
    float16_t outLayer[shader_io::kOutDim];
    [ForceUnroll]
    for (int j = 0; j < int(shader_io::kOutDim); ++j)
    {
        float16_t acc = mlpBuffer.Load<float16_t>(reconstructParams.mlpParams.b2 + uint(j) * 2u);
        const uint rowBytes = reconstructParams.mlpParams.w2 + uint(j) * uint(shader_io::kHiddenDim) * 2u;
        [ForceUnroll]
        for (int i = 0; i < int(shader_io::kHiddenDim); ++i)
        {
            acc += sharedBuffA[row * kCoopStride + uint(i)]
                 * mlpBuffer.Load<float16_t>(rowBytes + uint(i) * 2u);
        }
        outLayer[j] = clamp(acc, float16_t(0.0f), float16_t(1.0f));
    }
    return outLayer;
}

// Reference https://github.com/jeffbolznv/vk_cooperative_matrix_perf/blob/master/shaders/shmem.comp
[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID, uint groupIndex: SV_GroupIndex)
{
    uint screenWidth, screenHeight;
    outputImage.GetDimensions(screenWidth, screenHeight);

    const bool validPixel = (threadId.x < screenWidth) && (threadId.y < screenHeight);
    const float2 screenUV = validPixel
        ? (float2(threadId.xy) + 0.5f) / float2(float(screenWidth), float(screenHeight))
        : float2(0.0f, 0.0f);
    const float2 meshUV = gbuffer3.SampleLevel(screenUV, 0).xy;

    const float16_t s[shader_io::kOutDim] = MlpSample(meshUV, groupIndex);

    float output[shader_io::kOutDim];
    [ForceUnroll]
    for (int i = 0; i < int(shader_io::kOutDim); ++i)
    {
        output[i] = float(s[i]);
    }

    if (!validPixel)
    {
        return;
    }

    const float4 worldPosHit = gbuffer0.SampleLevel(screenUV, 0);
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