using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct TanhDerivativeKernel(ReadOnlyBuffer<float> outputs, ReadWriteBuffer<float> derivatives) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float y = outputs[i];
        derivatives[i] = 1f - y * y;
    }
}
