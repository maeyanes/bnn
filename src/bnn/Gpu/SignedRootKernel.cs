using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SignedRootKernel(ReadWriteBuffer<float> data) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float x = data[i];
        data[i] = Hlsl.Sign(x) * Hlsl.Sqrt(Hlsl.Abs(x));
    }
}
