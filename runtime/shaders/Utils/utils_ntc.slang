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

internal CoopVec<float16_t, OUT> HardGELULayer<let IN : int, let OUT : int>(const CoopVec<float16_t, IN> input,
                                                                            const ByteAddressBuffer mlp,
                                                                            const uint weightOffset,
                                                                            const uint biasOffset)
{
    // HardGELU(x):
    //   x <= -1.5        -> 0
    //   -1.5 < x < 1.5   -> x * (x + 1.5) / 3
    //   x >= 1.5         -> x
    // Branch-free, element-wise identity:
    //   HardGELU(x) = middle(clamp(x, -1.5, 1.5)) + max(x - 1.5, 0)
    const CoopVec<float16_t, OUT> x     = LinearLayer<IN, OUT>(input, mlp, weightOffset, biasOffset);
    const CoopVec<float16_t, OUT> cmp   = CoopVec<float16_t, OUT>(float16_t(1.5f));
    const CoopVec<float16_t, OUT> three = CoopVec<float16_t, OUT>(float16_t(3.0f));
    const CoopVec<float16_t, OUT> zero  = CoopVec<float16_t, OUT>(float16_t(0.0f));

    const CoopVec<float16_t, OUT> xc     = clamp(x, -cmp, cmp);
    const CoopVec<float16_t, OUT> middle = (xc * (xc + cmp)) / three;
    const CoopVec<float16_t, OUT> tail   = max(x - cmp, zero);
    return middle + tail;
}

// Hard coded becasue we know our MLP has 2 hidden layer and the in/out/hidden dims
public CoopVec<float16_t, OUT> MlpDecoder<let IN : int, let OUT : int, let HIDDEN : int>(
    const CoopVec<float16_t, IN> input, const ByteAddressBuffer mlp, const shader_io::MlpParams params)
{
    // Hidden layers
    // CoopVec<float16_t, HIDDEN> h0 = ReLULayer<IN, HIDDEN>(input, mlp, params.w0, params.b0);
    // CoopVec<float16_t, HIDDEN> h1 = ReLULayer<HIDDEN, HIDDEN>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, HIDDEN> h0 = HardGELULayer<IN, HIDDEN>(input, mlp, params.w0, params.b0);
    CoopVec<float16_t, HIDDEN> h1 = HardGELULayer<HIDDEN, HIDDEN>(h0, mlp, params.w1, params.b1);

    CoopVec<float16_t, OUT> raw = LinearLayer<HIDDEN, OUT>(h1, mlp, params.w2, params.b2);
    // return Sigmoid<OUT>(raw);
    return clamp(raw, CoopVec<float16_t, OUT>(float16_t(0.0f)), CoopVec<float16_t, OUT>(float16_t(1.0f)));
}
