using bnn.Data;
using bnn.Options;

namespace bnn.Services;

public class WeightsGeneratorService : IWeightsGeneratorService
{
    private Random? _rnd;

    /// <inheritdoc />
    public Weights GenerateWeights(InitWeightsOptions options) =>
        GenerateWeights(options.Input, options.Hidden, options.Output, options.Seed);

    /// <inheritdoc />
    public Weights GenerateWeights(int input,
                                   int hidden,
                                   int output,
                                   int seed)
    {
        _rnd = new Random(seed);

        Weights weights = Weights.CreateEmpty(input, hidden, output);

        GenerateCluster(weights.InitialCluster);
        GenerateCluster(weights.FinalCluster);

        return weights;
    }

    private void GenerateCluster(double[,] cluster)
    {
        for (int r = 0; r < cluster.GetLength(0); r++)
        {
            for (int c = 0; c < cluster.GetLength(1); c++)
            {
                cluster[r, c] = RandomWeight();
            }
        }
    }

    private double RandomWeight() => (_rnd!.NextDouble() - 0.5) / 10.0;
}