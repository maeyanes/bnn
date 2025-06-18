namespace bnn.Options;

public sealed class PredictOptions
{
    public string Activation { get; set; } = "sigmoid";

    public bool BinarizeOutput { get; set; } = false;

    public FileInfo DataFile { get; set; } = default!;

    public FileInfo? OutputFile { get; init; }

    public FileInfo WeightsFile { get; set; } = default!;

    public bool UseGpu { get; set; } = false;
}