using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct CubeRootKernel(ReadWriteBuffer<float> data) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float x = data[i];
        data[i] = Hlsl.Sign(x) * Hlsl.Pow(Hlsl.Abs(x), 1f / 3f);
    }
}
