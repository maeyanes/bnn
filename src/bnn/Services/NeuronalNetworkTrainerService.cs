using bnn.Data;
using bnn.Options;

namespace bnn.Services;

internal sealed class NeuralNetworkTrainerService : INeuralNetworkTrainerService
{
    private readonly string _currentDirectory = Directory.GetCurrentDirectory();

    public async Task<TrainingReport> TrainAsync(TrainOptions options,
                                                 TrainingData trainingData,
                                                 Weights initialWeights,
                                                 Func<double, double> activation,
                                                 Func<double, double> activationDerivative,
                                                 CancellationToken cancellationToken = default)
    {
        BackPropagationNeuralNetwork network = new(initialWeights, options.UseGpu);

        network.SetActivationFunction(activation, activationDerivative);

        TrainingReport trainingReport = network.BackPropagate(trainingData, options.LearningRate, options.MaxEpochs, options.Seed);

        int indexWidth = (int)Math.Ceiling(Math.Log10(options.MaxEpochs));

        OutputSettingsFile(options, indexWidth);

        for (int i = 0; i < trainingReport.ImprovedWeights.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            WeightsSnapshot snapshot = trainingReport.ImprovedWeights[i];

            Console.WriteLine($"{snapshot.Errors} errors at epoch {snapshot.Epoch:N0}");

            if (options.DisableImprovementWeights)
            {
                continue;
            }

            string filename = $"{options.OutputPrefix}{(i + 1).ToString().PadLeft(indexWidth, '0')}-{snapshot.Epoch}-{snapshot.Errors}.txt";

            FileInfo snapshotFile = new(Path.Combine(_currentDirectory, filename));

            await using FileStream fileStream = snapshotFile.Create();
            await using StreamWriter writer = new(fileStream);
            await writer.WriteAsync(snapshot.Weights.ToString());
        }

        Console.WriteLine();

        return trainingReport;
    }

    private void OutputSettingsFile(TrainOptions options, int indexWidth)
    {
        if (options.DisableImprovementWeights)
        {
            return;
        }

        string filename = $"{options.OutputPrefix}{0.ToString().PadLeft(indexWidth, '0')}-parameters.txt";

        FileInfo fileInfo = new(Path.Combine(_currentDirectory, filename));

        using FileStream fileStream = fileInfo.Create();
        using StreamWriter writer = new(fileStream);

        writer.WriteLine(options.WeightsFile is null ? $"Seed={options.Seed}" : $"WeightsFile={options.WeightsFile.FullName}");

        writer.WriteLine($"Hidden={options.Hidden}");
        writer.WriteLine($"LearningRate={options.LearningRate}");
        writer.WriteLine($"MaxEpochs={options.MaxEpochs}");
        writer.WriteLine($"Activation={options.Activation}");
    }
}