using ComputeSharp;

namespace bnn.Gpu;

public enum ActivationKind
{
    Sigmoid,
    Relu,
    Tanh,
    SignedRoot,
    CubeRoot
}

public static class ActivationFunctionsGpu
{
    public static ActivationKind ParseKind(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "sigmoid" => ActivationKind.Sigmoid,
            "relu" => ActivationKind.Relu,
            "tanh" => ActivationKind.Tanh,
            "signedroot" => ActivationKind.SignedRoot,
            "cuberoot" => ActivationKind.CubeRoot,
            _ => ActivationKind.Sigmoid
        };
    }

    public static void ApplyActivation(ReadWriteBuffer<float> buffer, int length, ActivationKind kind)
    {
        GraphicsDevice device = GraphicsDevice.GetDefault();

        switch (kind)
        {
            case ActivationKind.Sigmoid:
                device.For(length, new SigmoidKernel(buffer));
                break;
            case ActivationKind.Relu:
                device.For(length, new ReluKernel(buffer));
                break;
            case ActivationKind.Tanh:
                device.For(length, new TanhKernel(buffer));
                break;
            case ActivationKind.SignedRoot:
                device.For(length, new SignedRootKernel(buffer));
                break;
            case ActivationKind.CubeRoot:
                device.For(length, new CubeRootKernel(buffer));
                break;
        }
    }

    public static float[] Derivative(float[] outputs, ActivationKind kind)
    {
        using ReadOnlyBuffer<float> outputBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(outputs);
        using ReadWriteBuffer<float> derivativeBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(outputs.Length);
        GraphicsDevice device = GraphicsDevice.GetDefault();

        switch (kind)
        {
            case ActivationKind.Sigmoid:
                device.For(outputs.Length, new SigmoidDerivativeKernel(outputBuffer, derivativeBuffer));
                break;
            case ActivationKind.Relu:
                device.For(outputs.Length, new ReluDerivativeKernel(outputBuffer, derivativeBuffer));
                break;
            case ActivationKind.Tanh:
                device.For(outputs.Length, new TanhDerivativeKernel(outputBuffer, derivativeBuffer));
                break;
            case ActivationKind.SignedRoot:
                device.For(outputs.Length, new SignedRootDerivativeKernel(outputBuffer, derivativeBuffer));
                break;
            case ActivationKind.CubeRoot:
                device.For(outputs.Length, new CubeRootDerivativeKernel(outputBuffer, derivativeBuffer));
                break;
        }

        return derivativeBuffer.ToArray();
    }
}
