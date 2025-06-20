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

    public static void ComputeInitialErrorsGpu(double[,] finalCluster,
                                               double[] finalErrors,
                                               double[] hiddenLayer,
                                               double[] initialErrors,
                                               double[] outputLayer,
                                               double trainingRate,
                                               ActivationKind activation)
    {
        int hidden = hiddenLayer.Length;
        int outputs = finalErrors.Length;

        float[] clusterData = finalCluster.Cast<double>().Select(x => (float)x).ToArray();
        float[] finalErrorsData = Array.ConvertAll(finalErrors, x => (float)x);
        float[] hiddenDerivatives = ActivationFunctionsGpu.Derivative(Array.ConvertAll(hiddenLayer, x => (float)x), activation);
        float[] outputDerivatives = ActivationFunctionsGpu.Derivative(Array.ConvertAll(outputLayer, x => (float)x), activation);

        using ReadOnlyBuffer<float> clusterBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(clusterData);
        using ReadWriteBuffer<float> finalErrorsBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer(finalErrorsData);
        using ReadOnlyBuffer<float> hiddenDerBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(hiddenDerivatives);
        using ReadWriteBuffer<float> initialErrorsBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(hidden);

        int stride = finalCluster.GetLength(1);

        GraphicsDevice.GetDefault().For(hidden,
                                        new InitialErrorsKernel(clusterBuffer,
                                                               finalErrorsBuffer,
                                                               hiddenDerBuffer,
                                                               initialErrorsBuffer,
                                                               (float)trainingRate,
                                                               outputs,
                                                               stride));

        using ReadOnlyBuffer<float> outputDerBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(outputDerivatives);

        GraphicsDevice.GetDefault().For(outputs,
                                        new FinalErrorsKernel(finalErrorsBuffer,
                                                              outputDerBuffer,
                                                              (float)trainingRate));

        float[] initialGpu = initialErrorsBuffer.ToArray();
        float[] finalGpu = finalErrorsBuffer.ToArray();

        for (int i = 0; i < hidden; i++)
        {
            initialErrors[i] = initialGpu[i];
        }

        for (int i = 0; i < outputs; i++)
        {
            finalErrors[i] = finalGpu[i];
        }
    }

    public static void UpdateWeightsGpu(double[,] cluster, double[] inputs, double[] errors)
    {
        int rows = cluster.GetLength(0);
        int cols = cluster.GetLength(1);

        float[] clusterData = cluster.Cast<double>().Select(x => (float)x).ToArray();
        float[] inputData = Array.ConvertAll(inputs, x => (float)x);
        float[] errorData = Array.ConvertAll(errors, x => (float)x);

        using ReadWriteBuffer<float> clusterBuffer = GraphicsDevice.GetDefault().AllocateReadWriteBuffer(clusterData);
        using ReadOnlyBuffer<float> inputBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(inputData);
        using ReadOnlyBuffer<float> errorBuffer = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(errorData);

        GraphicsDevice.GetDefault().For(rows,
                                        new UpdateWeightsKernel(clusterBuffer,
                                                               inputBuffer,
                                                               errorBuffer,
                                                               cols - 1,
                                                               cols));

        float[] updated = clusterBuffer.ToArray();
        int index = 0;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                cluster[r, c] = updated[index++];
            }
        }
    }
}