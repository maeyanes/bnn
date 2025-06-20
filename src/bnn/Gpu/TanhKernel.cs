using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct TanhKernel(ReadWriteBuffer<float> data) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float x = data[i];
        data[i] = Hlsl.Tanh(x);
    }
}
