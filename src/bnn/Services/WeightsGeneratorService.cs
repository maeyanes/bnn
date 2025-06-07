using bnn.Data;
using bnn.Options;

namespace bnn.Services;

public class WeightsGeneratorService : IWeightsGeneratorService
{
    private Random? _rnd;

    /// <inheritdoc />
    public Weights GenerateWeights(InitWeightsOptions options)
    {
        _rnd = new Random(options.Seed);

        Weights weights = new(options.Input, options.Hidden, options.Output);

        GenerateCluster(weights.InitCluster);
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