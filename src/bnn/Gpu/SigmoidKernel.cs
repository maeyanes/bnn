using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SigmoidKernel(ReadWriteBuffer<float> data) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float x = data[i];
        data[i] = 1f / (1f + Hlsl.Exp(-x));
    }
}
