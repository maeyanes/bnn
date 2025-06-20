namespace bnn.Activation;

public static class ActivationFunctions
{
    public static readonly (Func<double, double> f, Func<double, double> df) CubeRoot =
        (x => Math.Sign(x) * Math.Pow(Math.Abs(x), 1.0 / 3.0), y => y == 0.0 ? 0.0 : 1.0 / (3.0 * y * y));

    public static readonly (Func<double, double> f, Func<double, double> df) ReLu = (x => Math.Max(0.0, x), y => y > 0.0 ? 1.0 : 0.0);

    public static readonly (Func<double, double> f, Func<double, double> df)
        Sigmoid = (x => 1.0 / (1.0 + Math.Exp(-x)), y => y * (1.0 - y));

    public static readonly (Func<double, double> f, Func<double, double> df) SignedRoot = (x => Math.Sign(x) * Math.Sqrt(Math.Abs(x)),
                                                                                           y => y == 0.0 ? 0.0 : 1.0 / (2.0 * y));

    public static readonly (Func<double, double> f, Func<double, double> df) Tanh = (Math.Tanh, y => 1.0 - y * y);
}