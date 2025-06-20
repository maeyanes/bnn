using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SignedRootDerivativeKernel(ReadOnlyBuffer<float> outputs, ReadWriteBuffer<float> derivatives) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float y = outputs[i];
        derivatives[i] = y == 0f ? 0f : 1f / (2f * y);
    }
}
