using bnn.Data;
using bnn.Options;

namespace bnn.Services;

public interface IWeightsGeneratorService
{
    public Weights GenerateWeights(InitWeightsOptions options);

    public Weights GenerateWeights(int input,
                                   int hidden,
                                   int output,
                                   int seed);
}