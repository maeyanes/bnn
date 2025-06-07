namespace bnn.Options;

public sealed class InitWeightsOptions
{
    public int Hidden { get; set; }

    public int Input { get; set; }

    public int Output { get; set; }

    public FileInfo OutputFile { get; set; } = default!;

    public int Seed { get; set; } = Environment.TickCount;
}