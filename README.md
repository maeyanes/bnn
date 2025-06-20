# Back Propagation Neural Network CLI Tools

This project provides a command line interface for training and using a simple back–propagation neural network. It is built with .NET 9 and exposes commands to generate weight files, train a model and produce predictions.

## Requirements

* .NET 9 SDK

## Building

Restore and build the project from the repository root:

```bash
 dotnet build bnn.sln
```

All executable code resides in `src/bnn`.

## Commands

Once built, you can run any command via `dotnet run`:

```bash
 dotnet run --project src/bnn -- <command> [options]
```

Available commands are:

### init-weights

Generate an initial weights file for a network.

```
--input, -i      Number of input values (required)
--hidden, -h     Number of hidden layer outputs (required)
--output, -o     Number of output values (required)
--seed, -s       Optional seed for deterministic generation
--outputFile     Path to the file where weights will be written (required)
```

Example:

```bash
 dotnet run --project src/bnn -- init-weights -i 2 -h 3 -o 1 --outputFile weights.txt
```

### train-network

Train a neural network using a dataset and an initial set of weights. The dataset file must start with the number of samples on the first line followed by a line with `<inputs> <outputs>`. Each subsequent line contains the input values followed by the expected output values for one sample.

```
--dataFile, -d            Training data file (required)
--weightsFile, -w         Initial weights file (optional)
--hidden, -h              Hidden layer size if no weights file is supplied
--maxEpochs, -e           Maximum training epochs (default: 1,000,000)
--learningRate, -l        Learning rate (default: 0.05)
--seed, -s                Random seed when generating weights
--activation, -a          Activation function: sigmoid | relu | tanh (default: sigmoid)
--outputPrefix            Prefix for the generated output files
--disableImprovementWeights  Do not save weights every time the network improves
```

### predict

Use a trained network to generate predictions for a set of inputs. The data file should contain whitespace separated numbers, one sample per line.

```
--dataFile, -d    Input data file (required)
--weightsFile, -w Trained weights file (required)
--outputFile, -o  Optional path to save raw output values
--binarizeOutput  Convert outputs >= 0.5 to 1 and others to 0
--activation, -a  Activation function: sigmoid | relu | tanh (default: sigmoid)
```

## Data Types

Some of the important types used by the commands are located under `src/bnn/Data`:

- `Weights` stores the network weights and is serialized using `WeightsSerializer`.
- `TrainingData` and `TrainingReport` describe the training process.
- `InputData` represents the matrix of samples consumed by the `predict` command.

## Next Steps

- Create training datasets and invoke `train-network` to produce weights.
- Use `predict` with new input data to evaluate the trained model.

