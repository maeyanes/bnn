namespace bnn.Data;

public sealed class TrainingReport
{
    public int EpochsExecuted { get; init; }

    public int? EpochZeroErrors { get; init; }

    public required List<WeightsSnapshot> ImprovedWeights { get; init; }

    public int MinErrors { get; init; }
}