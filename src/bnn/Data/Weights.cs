using System.Text;

namespace bnn.Data;

public sealed record Weights(int Input, int Hidden, int Output)
{
    public double[,] FinalCluster { get; } = new double[Output, Hidden + 1];

    public double[,] InitCluster { get; } = new double[Hidden, Input + 1];

    /// <inheritdoc />
    public override string ToString()
    {
        StringBuilder builder = new();

        builder.AppendLine($"{Input} {Hidden} {Output}");

        ClusterToString(builder, InitCluster);
        ClusterToString(builder, FinalCluster);

        return builder.ToString();
    }

    private static void ClusterToString(StringBuilder builder, double[,] cluster)
    {
        for (int r = 0; r < cluster.GetLength(0); r++)
        {
            for (int c = 0; c < cluster.GetLength(1); c++)
            {
                if (c > 0)
                {
                    builder.Append('\t');
                }

                builder.Append($"{cluster[r, c]}");
            }

            builder.AppendLine();
        }
    }
}