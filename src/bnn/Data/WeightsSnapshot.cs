namespace bnn.Data;

public sealed record WeightsSnapshot(int Epoch, int Errors, Weights Weights);