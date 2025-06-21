namespace bnn;

using bnn.Data;
using bnn.Gpu;

public interface IBackPropagationNeuralNetwork
{
    TrainingReport BackPropagate(TrainingData trainingData,
                                 double trainingRate,
                                 int maxEpoch,
                                 int seed);

    double[] Predict(double[] inputLayer);

    void SetActivationFunction(Func<double, double> activation,
                               Func<double, double> activationDerivative,
                               ActivationKind activationKind);
}
