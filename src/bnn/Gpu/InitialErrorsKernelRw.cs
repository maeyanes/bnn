using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct InitialErrorsKernelRw(
    ReadWriteBuffer<float> cluster,
    ReadWriteBuffer<float> finalErrors,
    ReadWriteBuffer<float> hiddenDerivatives,
    ReadWriteBuffer<float> initialErrors,
    float trainingRate,
    int outputs,
    int stride) : IComputeShader
{
    public void Execute()
    {
        int h = ThreadIds.X;
        float sum = 0f;

        for (int o = 0; o < outputs; o++)
        {
            sum += finalErrors[o] * cluster[o * stride + h];
        }

        initialErrors[h] = sum * trainingRate * hiddenDerivatives[h];
    }
}
