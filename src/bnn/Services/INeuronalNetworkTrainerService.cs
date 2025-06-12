using bnn.Data;
using bnn.Options;

namespace bnn.Services;

public interface INeuralNetworkTrainerService
{
    Task<TrainingReport> TrainAsync(TrainOptions options,
                                    TrainingData trainingData,
                                    Weights initialWeights,
                                    Func<double, double> activation,
                                    Func<double, double> activationDerivative,
                                    CancellationToken cancellationToken = default);
}