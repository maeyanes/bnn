using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct CubeRootDerivativeKernel(ReadOnlyBuffer<float> outputs, ReadWriteBuffer<float> derivatives) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float y = outputs[i];
        derivatives[i] = y == 0f ? 0f : 1f / (3f * y * y);
    }
}
