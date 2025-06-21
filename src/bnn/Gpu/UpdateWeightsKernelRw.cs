using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct UpdateWeightsKernelRw(
    ReadWriteBuffer<float> cluster,
    ReadWriteBuffer<float> inputs,
    ReadWriteBuffer<float> errors,
    int inputSize,
    int stride) : IComputeShader
{
    public void Execute()
    {
        int r = ThreadIds.X;
        float err = errors[r];

        for (int c = 0; c < inputSize; c++)
        {
            cluster[r * stride + c] += err * inputs[c];
        }

        cluster[r * stride + inputSize] += err;
    }
}
