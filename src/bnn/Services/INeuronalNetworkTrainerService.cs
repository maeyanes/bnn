using bnn.Data;
using bnn.Options;

namespace bnn.Services;

public interface INeuralNetworkTrainerService
{
    Task<TrainingReport> TrainAsync(TrainOptions options,
                                    TrainingData trainingData,
                                    Weights initialWeights,
                                    CancellationToken cancellationToken = default);
}