using ComputeSharp;

namespace bnn.Gpu;

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct FinalErrorsKernel(
    ReadWriteBuffer<float> finalErrors,
    ReadOnlyBuffer<float> outputDerivatives,
    float trainingRate) : IComputeShader
{
    public void Execute()
    {
        int o = ThreadIds.X;
        finalErrors[o] *= trainingRate * outputDerivatives[o];
    }
}
