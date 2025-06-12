using bnn.Data;
using bnn.Options;

namespace bnn.Services;

internal sealed class NeuralNetworkTrainerService : INeuralNetworkTrainerService
{
    public async Task<TrainingReport> TrainAsync(TrainOptions options,
                                                 TrainingData trainingData,
                                                 Weights initialWeights,
                                                 Func<double, double> activation,
                                                 Func<double, double> activationDerivative,
                                                 CancellationToken cancellationToken = default)
    {
        BackPropagationNeuralNetwork network = new(initialWeights);

        network.SetActivationFunction(activation, activationDerivative);

        TrainingReport trainingReport = network.BackPropagate(trainingData, options.LearningRate, options.MaxEpochs, options.Seed);

        int indexWidth = (int)Math.Ceiling(Math.Log10(options.MaxEpochs));

        for (int i = 0; i < trainingReport.ImprovedWeights.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            WeightsSnapshot snapshot = trainingReport.ImprovedWeights[i];

            Console.WriteLine($"{snapshot.Errors} errors at epoch {snapshot.Epoch:N0}");

            if (options.DisableImprovementWeights)
            {
                continue;
            }

            string filename = $"{options.OutputPrefix}{i.ToString().PadLeft(indexWidth, '0')}-{snapshot.Epoch}-{snapshot.Errors}.txt";

            FileInfo snapshotFile = new(Path.Combine(options.DataFile.DirectoryName!, filename));

            await using FileStream fileStream = snapshotFile.Create();
            await using StreamWriter writer = new(fileStream);
            await writer.WriteAsync(snapshot.Weights.ToString());
        }

        Console.WriteLine();

        return trainingReport;
    }
}