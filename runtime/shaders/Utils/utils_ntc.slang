implementing Utils;

import ShaderIO;

internal CoopVec<float16_t, OUT> LinearLayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                          const ByteAddressBuffer mlp,
                                                                          const uint weightOffset,
                                                                          const uint biaisOffset)
{
    return coopVecMatMulAddPacked<float16_t, OUT>(
        input, CoopVecComponentType::Float16, IN, mlp, weightOffset, CoopVecComponentType::Float16, mlp, biaisOffset,
        CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false, IN * 2);
}

internal CoopVec<float16_t, OUT> ReLULayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                        const ByteAddressBuffer mlp,
                                                                        const uint weightOffset, const uint biasOffset)
{
    return max(LinearLayer<IN, OUT>(input, mlp, weightOffset, biasOffset), CoopVec<float16_t, OUT>(float16_t(0.0f)));
}

internal CoopVec<float16_t, N> Sigmoid<let N : int>(const CoopVec<float16_t, N> input)
{
    const CoopVec<float16_t, N> c1 = CoopVec<float16_t, N>(float16_t(1.0f));
    return c1 /
           (c1 + exp(-clamp(input, CoopVec<float16_t, N>(float16_t(-10.0f)), CoopVec<float16_t, N>(float16_t(10.0f)))));
}

// Hard coded becasue we know our MLP has 2 hidden layer and the in/out/hidden dims
public CoopVec<float16_t, OUT> MlpDecoder<let IN : int, let OUT : int, let HIDDEN : int>(
    const CoopVec<float16_t, IN> input, const ByteAddressBuffer mlp, const shader_io::MlpParams params)
{
    // Hidden layers
    CoopVec<float16_t, HIDDEN> h0 = ReLULayer<IN, HIDDEN>(input, mlp, params.w0, params.b0);
    CoopVec<float16_t, HIDDEN> h1 = ReLULayer<HIDDEN, HIDDEN>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, OUT> raw = LinearLayer<HIDDEN, OUT>(h1, mlp, params.w2, params.b2);
    return Sigmoid<OUT>(raw);
    // return clamp(raw, CoopVec<float16_t, OUT>(float16_t(0.0f)), CoopVec<float16_t, OUT>(float16_t(1.0f)));
}
