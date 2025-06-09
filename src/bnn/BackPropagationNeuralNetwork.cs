using System.Diagnostics.CodeAnalysis;
using bnn.Data;

namespace bnn;

public sealed class BackPropagationNeuralNetwork
{
    private readonly double[,] _finalCluster;
    private readonly int _hidden;
    private readonly double[,] _initialCluster;
    private readonly int _inputs;
    private readonly int _outputs;
    private double[] _hiddenLayer;

    public BackPropagationNeuralNetwork(int inputs = 0, int hidden = 0, int outputs = 0)

    {
        _inputs = inputs;
        _hidden = hidden;
        _outputs = outputs;

        _initialCluster = new double[_hidden, _inputs + 1];
        _finalCluster = new double[_outputs, _hidden + 1];
        _hiddenLayer = new double[_hidden];
    }

    public BackPropagationNeuralNetwork(Weights weights)
    {
        _inputs = weights.Input;
        _hidden = weights.Hidden;
        _outputs = weights.Output;

        _initialCluster = new double[_hidden, _inputs + 1];
        _finalCluster = new double[_outputs, _hidden + 1];
        _hiddenLayer = new double[_hidden];

        // Copiar los pesos desde el objeto Weights a las matrices internas
        Array.Copy(weights.InitialCluster, _initialCluster, weights.InitialCluster.Length);
        Array.Copy(weights.FinalCluster, _finalCluster, weights.FinalCluster.Length);
    }

    public Func<double, double> ActivationDerivate { get; set; } = x => x * (1.0 - x);

    public Func<double, double> ActivationFunction { get; set; } = x => 1.0 / (1.0 + Math.Exp(-x));

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

        for (int epoch = 0; epoch < maxEpoch; epoch++)
        {
            int mistakesPerEpoch = 0;

            for (int sampleIndex = 0; sampleIndex < trainingData.Samples; sampleIndex++)
            {
                double[]? input = trainingData.GetInput(sampleIndex);
                double[] outputLayer = Pass(input);

                int mistakesPerOutput = CalculateOutputErrors(trainingData, sampleIndex, outputLayer, finalErrors);

                if (mistakesPerOutput > 0)
                {
                    mistakesPerEpoch++;
                }

                ComputeInitialErrors(finalErrors, initialErrors, outputLayer, trainingRate);

                UpdateWeights(_initialCluster, input, initialErrors);

                UpdateWeights(_finalCluster, _hiddenLayer, finalErrors);
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

        return new TrainingReport
               {
                   EpochsExecuted = epochZeroErrors.HasValue ? epochZeroErrors.Value + 1 : maxEpoch,
                   EpochZeroErrors = epochZeroErrors,
                   MinErrors = minMistakesPerEpoch,
                   ImprovedWeights = improvedWeights
               };
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

    private double Activate(double x) => ActivationFunction(x);

    private double[] CalculateLayerOutput(double[] inputs, double[,] cluster)
    {
        double[] outputs = new double[cluster.GetLength(0)];

        for (int i = 0; i < cluster.GetLength(0); i++)
        {
            double neuronOutput = cluster[i, inputs.Length] + inputs.Select((t, j) => t * cluster[i, j]).Sum();

            outputs[i] = Activate(neuronOutput);
        }

        return outputs;
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

            initialErrors[h] *= trainingRate * ActivationDerivate(_hiddenLayer[h]);
        }

        for (int o = 0; o < _outputs; o++)
        {
            finalErrors[o] *= trainingRate * ActivationDerivate(outputLayer[o]);
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