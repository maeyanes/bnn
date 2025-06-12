namespace bnn.Options;

public sealed class TrainOptions
{
    public string Activation { get; set; } = "sigmoid";

    public FileInfo DataFile { get; set; } = default!;

    public bool DisableImprovementWeights { get; set; } = false;

    public int Hidden { get; set; } = 1;

    public double LearningRate { get; set; } = 0.05;

    public int MaxEpochs { get; set; } = 1000000;

    public string OutputPrefix { get; set; } = "W-";

    public int Seed { get; set; } = Environment.TickCount;

    public FileInfo WeightsFile { get; set; } = default!;
}