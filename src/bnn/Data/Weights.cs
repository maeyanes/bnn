using bnn.Serialization;

namespace bnn.Data;

public sealed record Weights
{
    public Weights(int input,
                   int hidden,
                   int output,
                   double[,] initialCluster,
                   double[,] finalCluster)
    {
        if (initialCluster.GetLength(0) != hidden || initialCluster.GetLength(1) != input + 1)
        {
            throw new ArgumentException("InitialCluster dimensions must be [Hidden, Input + 1]");
        }

        if (finalCluster.GetLength(0) != output || finalCluster.GetLength(1) != hidden + 1)
        {
            throw new ArgumentException("FinalCluster dimensions must be [Output, Hidden + 1]");
        }

        Input = input;
        Hidden = hidden;
        Output = output;

        InitialCluster = (double[,])initialCluster.Clone();
        FinalCluster = (double[,])finalCluster.Clone();
    }

    public double[,] FinalCluster { get; }

    public int Hidden { get; }

    public double[,] InitialCluster { get; }

    public int Input { get; }

    public int Output { get; }

    public static Weights CreateEmpty(int input, int hidden, int output) =>
        new(input,
            hidden,
            output,
            new double[hidden, input + 1],
            new double[output, hidden + 1]);

    /// <inheritdoc />
    public override string ToString() => WeightsSerializer.Serialize(this);
}