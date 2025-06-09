using bnn.Extensions;

namespace bnn.Data;

public sealed record TrainingData(int Samples = 0, int Inputs = 0, int Outputs = 0)
{
    public double[,] InputData { get; } = new double[Samples, Inputs];

    public double[,] OutputData { get; } = new double[Samples, Outputs];

    public double[] GetInput(int sampleIndex) => InputData.GetRow(sampleIndex);

    public double[] GetOutput(int sampleIndex) => OutputData.GetRow(sampleIndex);
}