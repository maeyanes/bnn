using ComputeSharp;

namespace bnn.Gpu;

public static class GpuBackpropagation
{
    public static double[] CalculateLayerOutputGpu(double[] inputs, double[,] weights, Func<double, double> activation)
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

        float[] output = outputBuffer.ToArray();

        return output.Select(x => activation(x)).ToArray();
    }

    public static void ComputeInitialErrorsGpu(double[,] finalCluster,
                                               double[] finalErrors,
                                               double[] hiddenLayer,
                                               double[] initialErrors,
                                               double[] outputLayer,
                                               double trainingRate,
                                               Func<double, double> activationDerivative)
    {
        int hidden = hiddenLayer.Length;
        int outputs = finalErrors.Length;

        Parallel.For(0,
                     hidden,
                     h =>
                     {
                         double sum = 0;

                         for (int o = 0; o < outputs; o++)
                         {
                             sum += finalErrors[o] * finalCluster[o, h];
                         }

                         initialErrors[h] = sum * trainingRate * activationDerivative(hiddenLayer[h]);
                     });

        Parallel.For(0, outputs, o => { finalErrors[o] *= trainingRate * activationDerivative(outputLayer[o]); });
    }

    public static void UpdateWeightsGpu(double[,] cluster, double[] inputs, double[] errors)
    {
        int rows = cluster.GetLength(0);
        int cols = cluster.GetLength(1);

        Parallel.For(0,
                     rows,
                     r =>
                     {
                         for (int c = 0; c < cols - 1; c++)
                         {
                             cluster[r, c] += errors[r] * inputs[c];
                         }

                         cluster[r, cols - 1] += errors[r];
                     });
    }
}