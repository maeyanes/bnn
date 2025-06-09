﻿using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using bnn.Data;
using bnn.Options;
using bnn.Serialization;
using bnn.Services;
using bnn.Utils;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace bnn.Commands;

internal static class TrainNetworkCommand
{
    public static Command Create()
    {
        Command cmd = new("train-network", "Train a neural network using the specified dataset and weights.");

        cmd.AddOption(new Option<FileInfo>(["--dataFile", "-d"], "Path to the training dataset file") { IsRequired = true });
        cmd.AddOption(new Option<FileInfo>(["--weightsFile", "-w"],
                                           "Path to the initial weights file. If omitted, random weights will be generated.")
                      {
                          IsRequired = false
                      });
        cmd.AddOption(new Option<int>(["--hidden", "-h"],
                                      "Number of outputs in the hidden layer. If a weights file is provided and this value is not set, the value from the file will be used. If neither is provided, a default value of 1 is used.")
                      {
                          IsRequired = false
                      });
        cmd.AddOption(new Option<int>(["--maxEpochs", "-e"], "Maximum number of training epochs (iterations over the entire dataset).")
                      {
                          IsRequired = false
                      });
        cmd.AddOption(new Option<double>(["--learningRate", "-l"],
                                         "Learning rate used during training (typically a small value like 0.01).") { IsRequired = false });
        cmd.AddOption(new Option<bool>("--outputImprovementWeights",
                                       "If enabled, outputs the weights to a file every time the model improves.") { IsRequired = false });
        cmd.AddOption(new Option<int>(["--seed", "-s"],
                                      "Optional random seed to initialize weights (only used if no weights file is provided).")
                      {
                          IsRequired = false
                      });
        cmd.AddOption(new Option<string>("--prefix",
                                         "Optional prefix for naming output files. If not specified, a default prefix will be used."));

        cmd.Handler = CommandHandler.Create<TrainOptions, IHost>(Run);

        return cmd;
    }

    private static TrainingData LoadTrainingData(FileInfo dataFile)
    {
        if (!dataFile.Exists)
        {
            throw new ArgumentException("The specified training data file does not exist.", nameof(dataFile));
        }

        using FileStream fileStream = dataFile.OpenRead();
        using TextReader reader = new StreamReader(fileStream);

        string? line = reader.ReadLine();

        if (line is null || !int.TryParse(line.Trim(), out int samples) || samples <= 0)
        {
            throw new InvalidDataException("Invalid or missing number of training samples.");
        }

        line = reader.ReadLine();

        if (line is null)
        {
            throw new InvalidDataException("Missing input/output definition line.");
        }

        string[] inputOutput = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

        if (inputOutput.Length != 2 || !int.TryParse(inputOutput[0], out int inputs) || !int.TryParse(inputOutput[1], out int outputs) ||
            inputs <= 0 || outputs <= 0)
        {
            throw new InvalidDataException("Invalid input/output count format. Expected: <inputs> <outputs>");
        }

        TrainingData trainingData = new(samples, inputs, outputs);

        for (int s = 0; s < samples; s++)
        {
            line = reader.ReadLine();

            if (line is null)
            {
                throw new InvalidDataException($"Expected {samples} training data rows, but found only {s}.");
            }

            string[] values = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

            if (values.Length != inputs + outputs)
            {
                throw new InvalidDataException($"Training sample at line {s + 3} has {values.Length} values, expected {inputs + outputs}.");
            }

            for (int i = 0; i < inputs; i++)
            {
                if (!double.TryParse(values[i], out double inputValue))
                {
                    throw new InvalidDataException($"Invalid input value '{values[i]}' at line {s + 3}, column {i + 1}.");
                }

                trainingData.InputData[s, i] = inputValue;
            }

            for (int o = 0; o < outputs; o++)
            {
                int index = outputs - inputs + o;

                if (!double.TryParse(values[index], out double outputValue))
                {
                    throw new InvalidDataException($"Invalid output value '{values[index]}' at line {s + 3}, column {index + 1}.");
                }

                trainingData.OutputData[s, o] = outputValue;
            }
        }


        return trainingData;
    }

    private static async Task Run(TrainOptions options, IHost host)
    {
        try
        {
            ConsoleOutput.PrintInfo($"Training with a max of {options.MaxEpochs:N0} epochs at learning rate {options.LearningRate}");

            TrainingData trainingData = LoadTrainingData(options.DataFile);

            ConsoleOutput.PrintInfo($"Training data: {trainingData.Samples:N0} samples, {trainingData.Inputs:N0} inputs, {trainingData.Outputs:N0} outputs.");

            Weights initialWeights;

            if (options.WeightsFile is not null)
            {
                ConsoleOutput.PrintInfo($"Initial weights loaded from file: {options.WeightsFile.FullName}.");

                initialWeights = WeightsSerializer.DeserializeFromFile(options.WeightsFile);
            }
            else
            {
                ConsoleOutput.PrintInfo($"Initial weights generated using seed {options.Seed}");

                IWeightsGeneratorService generator = host.Services.GetRequiredService<IWeightsGeneratorService>();

                initialWeights = generator.GenerateWeights(trainingData.Inputs, options.Hidden, trainingData.Outputs, options.Seed);
            }

            Console.WriteLine();

            INeuralNetworkTrainerService trainer = host.Services.GetRequiredService<INeuralNetworkTrainerService>();

            TrainingReport trainingReport = await trainer.TrainAsync(options, trainingData, initialWeights);

            Console.WriteLine();

            if (trainingReport.EpochZeroErrors.HasValue)
            {
                ConsoleOutput.PrintSuccess($"Network trained successfully in {trainingReport.EpochZeroErrors:N0} epochs.");
            }
            else
            {
                await ConsoleOutput
                    .PrintErrorAsync($"Network training completed after {trainingReport.EpochsExecuted:N0} epochs with {trainingReport.MinErrors} errors.",
                                     false);
            }
        }
        catch (Exception ex)
        {
            await ConsoleOutput.PrintErrorAsync($"An error occurred while training the network: {ex.Message}");

            Environment.ExitCode = ex.HResult;
        }
    }
}