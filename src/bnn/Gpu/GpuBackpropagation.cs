using ComputeSharp;

namespace bnn.Gpu;

public static class GpuBackpropagation
{
    public static double[] CalculateLayerOutputGpu(double[] inputs, double[,] weights, ActivationKind activation)
    {
        int outputSize = weights.GetLength(0);
        int inputSize = inputs.Length;

        using ReadOnlyBuffer<float>
            inputBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(inputs.Select(x => (float)x).ToArray());
        using ReadOnlyBuffer<float> weightBuffer = GraphicsDevice.GetDefault()
                                                                 .AllocateReadOnlyBuffer(weights.Cast<double>()
                                                                                             .Select(x => (float)x)
                                                                                             .ToArray());
        using ReadWriteBuffer<float> outputBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(outputSize);

        int stride = inputSize + 1;

        GraphicsDevice.GetDefault()
        .For(outputSize,
             new Kernel(inputBuffer,
                        weightBuffer,
                        outputBuffer,
                        inputSize,
                        stride));

        ActivationFunctionsGpu.ApplyActivation(outputBuffer, outputSize, activation);

        float[] output = outputBuffer.ToArray();

        return Array.ConvertAll(output, x => (double)x);
    }

    public static void CalculateLayerOutputGpu(ReadWriteBuffer<float> input,
                                               ReadWriteBuffer<float> weights,
                                               ReadWriteBuffer<float> output,
                                               int inputSize,
                                               int stride,
                                               ActivationKind activation)
    {
        GraphicsDevice.GetDefault().For(output.Length,
                                        new KernelRw(input,
                                                     weights,
                                                     output,
                                                     inputSize,
                                                     stride));

        ActivationFunctionsGpu.ApplyActivation(output, output.Length, activation);
    }

    public static void ComputeInitialErrorsGpu(ReadWriteBuffer<float> finalCluster,
                                               ReadWriteBuffer<float> finalErrors,
                                               ReadWriteBuffer<float> hiddenLayer,
                                               ReadWriteBuffer<float> initialErrors,
                                               ReadWriteBuffer<float> outputLayer,
                                               float trainingRate,
                                               ActivationKind activation,
                                               int outputs,
                                               int stride)
    {
        int hidden = hiddenLayer.Length;

        using ReadWriteBuffer<float> hiddenDerBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(hidden);
        using ReadWriteBuffer<float> outputDerBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(outputs);

        ActivationFunctionsGpu.Derivative(hiddenLayer, hiddenDerBuffer, activation);
        ActivationFunctionsGpu.Derivative(outputLayer, outputDerBuffer, activation);

        GraphicsDevice.GetDefault().For(hidden,
                                        new InitialErrorsKernelRw(finalCluster,
                                                                 finalErrors,
                                                                 hiddenDerBuffer,
                                                                 initialErrors,
                                                                 (float)trainingRate,
                                                                 outputs,
                                                                 stride));

        GraphicsDevice.GetDefault().For(outputs,
                                        new FinalErrorsKernel(finalErrors,
                                                              outputDerBuffer,
                                                              (float)trainingRate));
    }

    public static void UpdateWeightsGpu(ReadWriteBuffer<float> cluster,
                                        ReadWriteBuffer<float> inputs,
                                        ReadWriteBuffer<float> errors,
                                        int inputSize,
                                        int stride)
    {
        int rows = errors.Length;

        GraphicsDevice.GetDefault().For(rows,
                                        new UpdateWeightsKernelRw(cluster,
                                                                 inputs,
                                                                 errors,
                                                                 inputSize,
                                                                 stride));
    }
}