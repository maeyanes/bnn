using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct ReluKernel(ReadWriteBuffer<float> data) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float x = data[i];
        data[i] = Hlsl.Max(0f, x);
    }
}
