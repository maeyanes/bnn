using System.Diagnostics.CodeAnalysis;
using bnn.Activation;
using bnn.Data;
using bnn.Gpu;
using ComputeSharp;

namespace bnn;

public sealed class BackPropagationNeuralNetworkGpu : IBackPropagationNeuralNetwork
{
    private readonly double[,] _finalCluster;
    private readonly int _hidden;
    private readonly double[,] _initialCluster;
    private readonly int _inputs;
    private readonly int _outputs;
    private Func<double, double> _activationDerivative;
    private Func<double, double> _activationFunction;
    private ActivationKind _activationKind;
    private double[] _hiddenLayer;

    public BackPropagationNeuralNetworkGpu(int inputs = 0,
                                            int hidden = 0,
                                            int outputs = 0)

    {
        _inputs = inputs;
        _hidden = hidden;
        _outputs = outputs;

        _initialCluster = new double[_hidden, _inputs + 1];
        _finalCluster = new double[_outputs, _hidden + 1];
        _hiddenLayer = new double[_hidden];

        (Func<double, double> activation, Func<double, double> activationDerivative) = ActivationFunctions.Sigmoid;

        _activationFunction = activation;
        _activationDerivative = activationDerivative;
        _activationKind = ActivationKind.Sigmoid;
    }

    public BackPropagationNeuralNetworkGpu(Weights weights) : this(weights.Input, weights.Hidden, weights.Output)
    {
        // Copiar los pesos desde el objeto Weights a las matrices internas
        Array.Copy(weights.InitialCluster, _initialCluster, weights.InitialCluster.Length);
        Array.Copy(weights.FinalCluster, _finalCluster, weights.FinalCluster.Length);
    }

    public TrainingReport BackPropagate(TrainingData trainingData,
                                        double trainingRate,
                                        int maxEpoch,
                                        int seed)
    {
        if (trainingData.Inputs != _inputs || trainingData.Outputs != _outputs)
        {
            throw new ArgumentException("Input or output dimensions of the training data do not match the network configuration.");
        }

        double[] initialErrors = new double[_hidden];
        double[] finalErrors = new double[_outputs];
        int minMistakesPerEpoch = trainingData.Samples + 1;
        int? epochZeroErrors = null;
        List<WeightsSnapshot>? improvedWeights = [];

        GraphicsDevice device = GraphicsDevice.GetDefault();

        using ReadWriteBuffer<float> initialClusterBuffer = device.AllocateReadWriteBuffer(_initialCluster.Cast<double>()
                                                                                                   .Select(x => (float)x)
                                                                                                   .ToArray());
        using ReadWriteBuffer<float> finalClusterBuffer = device.AllocateReadWriteBuffer(_finalCluster.Cast<double>()
                                                                                                 .Select(x => (float)x)
                                                                                                 .ToArray());
        using ReadWriteBuffer<float> hiddenLayerBuffer = device.AllocateReadWriteBuffer<float>(_hidden);
        using ReadWriteBuffer<float> outputLayerBuffer = device.AllocateReadWriteBuffer<float>(_outputs);
        using ReadWriteBuffer<float> initialErrorsBuffer = device.AllocateReadWriteBuffer<float>(_hidden);
        using ReadWriteBuffer<float> finalErrorsBuffer = device.AllocateReadWriteBuffer<float>(_outputs);

        for (int epoch = 0; epoch < maxEpoch; epoch++)
        {
            int mistakesPerEpoch = 0;

            for (int sampleIndex = 0; sampleIndex < trainingData.Samples; sampleIndex++)
            {
                double[]? input = trainingData.GetInput(sampleIndex);

                using ReadWriteBuffer<float> inputBuffer = device.AllocateReadWriteBuffer(input.Select(x => (float)x).ToArray());

                GpuBackpropagation.CalculateLayerOutputGpu(inputBuffer,
                                                           initialClusterBuffer,
                                                           hiddenLayerBuffer,
                                                           _inputs,
                                                           _inputs + 1,
                                                           _activationKind);

                GpuBackpropagation.CalculateLayerOutputGpu(hiddenLayerBuffer,
                                                           finalClusterBuffer,
                                                           outputLayerBuffer,
                                                           _hidden,
                                                           _hidden + 1,
                                                           _activationKind);

                double[] outputLayer = Array.ConvertAll(outputLayerBuffer.ToArray(), x => (double)x);

                int mistakesPerOutput = CalculateOutputErrors(trainingData, sampleIndex, outputLayer, finalErrors);

                if (mistakesPerOutput > 0)
                {
                    mistakesPerEpoch++;
                }

                using ReadWriteBuffer<float> finalErrBuffer = device.AllocateReadWriteBuffer(finalErrors.Select(x => (float)x).ToArray());

                GpuBackpropagation.ComputeInitialErrorsGpu(finalClusterBuffer,
                                                           finalErrBuffer,
                                                           hiddenLayerBuffer,
                                                           initialErrorsBuffer,
                                                           outputLayerBuffer,
                                                           trainingRate,
                                                           _activationKind,
                                                           _outputs,
                                                           _hidden + 1);

                GpuBackpropagation.UpdateWeightsGpu(initialClusterBuffer,
                                                   inputBuffer,
                                                   initialErrorsBuffer,
                                                   _inputs,
                                                   _inputs + 1);

                GpuBackpropagation.UpdateWeightsGpu(finalClusterBuffer,
                                                   hiddenLayerBuffer,
                                                   finalErrBuffer,
                                                   _hidden,
                                                   _hidden + 1);
            }

            if (mistakesPerEpoch < minMistakesPerEpoch)
            {
                minMistakesPerEpoch = mistakesPerEpoch;

                improvedWeights.Add(CreateSnapshotWeights(epoch, mistakesPerEpoch));
            }

            if (mistakesPerEpoch != 0 || epochZeroErrors is not null)
            {
                continue;
            }


            epochZeroErrors = epoch;

            break;
        }

        float[] initialGpu = initialClusterBuffer.ToArray();
        float[] finalGpu = finalClusterBuffer.ToArray();

        int index = 0;

        for (int r = 0; r < _hidden; r++)
        {
            for (int c = 0; c <= _inputs; c++)
            {
                _initialCluster[r, c] = initialGpu[index++];
            }
        }

        index = 0;

        for (int r = 0; r < _outputs; r++)
        {
            for (int c = 0; c <= _hidden; c++)
            {
                _finalCluster[r, c] = finalGpu[index++];
            }
        }

        return new TrainingReport
               {
                   EpochsExecuted = epochZeroErrors.HasValue ? epochZeroErrors.Value + 1 : maxEpoch,
                   EpochZeroErrors = epochZeroErrors,
                   MinErrors = minMistakesPerEpoch,
                   ImprovedWeights = improvedWeights
               };
    }

    public double[] Predict(double[] inputLayer)
    {
        if (inputLayer.Length != _inputs)
        {
            throw new ArgumentException($"Input layer must have {_inputs} elements, but received {inputLayer.Length}.");
        }

        GraphicsDevice device = GraphicsDevice.GetDefault();

        using ReadWriteBuffer<float> inputBuffer = device.AllocateReadWriteBuffer(inputLayer.Select(x => (float)x).ToArray());
        using ReadWriteBuffer<float> initBuffer = device.AllocateReadWriteBuffer(_initialCluster.Cast<double>().Select(x => (float)x).ToArray());
        using ReadWriteBuffer<float> finalBuffer = device.AllocateReadWriteBuffer(_finalCluster.Cast<double>().Select(x => (float)x).ToArray());
        using ReadWriteBuffer<float> hiddenBuffer = device.AllocateReadWriteBuffer<float>(_hidden);
        using ReadWriteBuffer<float> outputBuffer = device.AllocateReadWriteBuffer<float>(_outputs);

        GpuBackpropagation.CalculateLayerOutputGpu(inputBuffer,
                                                   initBuffer,
                                                   hiddenBuffer,
                                                   _inputs,
                                                   _inputs + 1,
                                                   _activationKind);

        GpuBackpropagation.CalculateLayerOutputGpu(hiddenBuffer,
                                                   finalBuffer,
                                                   outputBuffer,
                                                   _hidden,
                                                   _hidden + 1,
                                                   _activationKind);

        float[] result = outputBuffer.ToArray();

        return Array.ConvertAll(result, x => (double)x);
    }

    public void SetActivationFunction(Func<double, double> activation,
                                      Func<double, double> activationDerivative,
                                      ActivationKind activationKind)
    {
        _activationFunction = activation;
        _activationDerivative = activationDerivative;
        _activationKind = activationKind;
    }

    private static void UpdateWeights(double[,] cluster, double[] inputs, double[] errors)
    {
        int rows = cluster.GetLength(0);
        int cols = cluster.GetLength(1) - 1;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                cluster[r, c] += errors[r] * inputs[c];
            }

            // Actualiza el bias
            cluster[r, cols] += errors[r];
        }
    }

    private double Activate(double x) => _activationFunction(x);

    private double[] CalculateLayerOutput(double[] inputs, double[,] cluster)
    {
        return GpuBackpropagation.CalculateLayerOutputGpu(inputs, cluster, _activationKind);
    }

    [SuppressMessage("ReSharper", "CompareOfFloatsByEqualityOperator")]
    private int CalculateOutputErrors(TrainingData data,
                                      int sampleIndex,
                                      double[] outputLayer,
                                      double[] finalErrors)
    {
        int mistakes = 0;

        for (int i = 0; i < _outputs; i++)
        {
            if (double.IsNaN(outputLayer[i]) || double.IsInfinity(outputLayer[i]))
            {
                throw new InvalidOperationException($"Invalid output detected at neuron {i}: {outputLayer[i]}");
            }

            finalErrors[i] = data.OutputData[sampleIndex, i] - outputLayer[i];

            double result = outputLayer[i] >= 0.5 ? 1.0 : 0.0;

            if (result != data.OutputData[sampleIndex, i])
            {
                mistakes++;
            }
        }

        return mistakes;
    }

    private void ComputeInitialErrors(double[] finalErrors,
                                      double[] initialErrors,
                                      double[] outputLayer,
                                      double trainingRate)
    {
        for (int h = 0; h < _hidden; h++)
        {
            initialErrors[h] = 0.0;

            for (int o = 0; o < _outputs; o++)
            {
                initialErrors[h] += finalErrors[o] * _finalCluster[o, h];
            }

            initialErrors[h] *= trainingRate * _activationDerivative(_hiddenLayer[h]);
        }

        for (int o = 0; o < _outputs; o++)
        {
            finalErrors[o] *= trainingRate * _activationDerivative(outputLayer[o]);
        }
    }

    private WeightsSnapshot CreateSnapshotWeights(int epoch, int errors)
    {
        Weights snapshot = Weights.CreateEmpty(_inputs, _hidden, _outputs);

        Array.Copy(_initialCluster, snapshot.InitialCluster, _initialCluster.Length);
        Array.Copy(_finalCluster, snapshot.FinalCluster, _finalCluster.Length);

        return new WeightsSnapshot(epoch, errors, snapshot);
    }

    private double[] Pass(double[] inputLayer)
    {
        _hiddenLayer = CalculateLayerOutput(inputLayer, _initialCluster);

        return CalculateLayerOutput(_hiddenLayer, _finalCluster);
    }
}
