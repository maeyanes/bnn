using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct KernelRw(
    ReadWriteBuffer<float> input,
    ReadWriteBuffer<float> weights,
    ReadWriteBuffer<float> output,
    int inputSize,
    int stride) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = weights[i * stride + inputSize];

        for (int j = 0; j < inputSize; j++)
        {
            sum += input[j] * weights[i * stride + j];
        }

        output[i] = sum;
    }
}
