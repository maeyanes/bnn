using bnn.Data;
using bnn.Gpu;
using bnn.Options;

namespace bnn.Services;

public interface INeuralNetworkTrainerService
{
    Task<TrainingReport> TrainAsync(TrainOptions options,
                                    TrainingData trainingData,
                                    Weights initialWeights,
                                     Func<double, double> activation,
                                     Func<double, double> activationDerivative,
                                     ActivationKind activationKind,
                                     CancellationToken cancellationToken = default);
}