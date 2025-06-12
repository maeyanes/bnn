namespace bnn.Activation;

public static class ActivationFunctions
{
    public static readonly (Func<double, double> f, Func<double, double> df) ReLu = (x => Math.Max(0.0, x),
                                                                                     y => y > 0.0 ? 1.0 : 0.0 // y = ReLU(x)
                                                                                    );

    public static readonly (Func<double, double> f, Func<double, double> df) Sigmoid = (x => 1.0 / (1.0 + Math.Exp(-x)),
                                                                                        y => y * (1.0 - y) // y = sigmoid(x)
                                                                                       );

    public static readonly (Func<double, double> f, Func<double, double> df) Tanh = (Math.Tanh, y => 1.0 - y * y // y = tanh(x)
                                                                                    );
}