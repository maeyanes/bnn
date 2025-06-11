using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using bnn.Data;
using bnn.Extensions;
using bnn.Options;
using bnn.Serialization;
using bnn.Utils;
using Microsoft.Extensions.Hosting;

namespace bnn.Commands;

public static class PredictCommand
{
    public static Command Create()
    {
        Command cmd = new("predict", "Generates predictions using a trained neural network and a dataset.");

        cmd.AddOption(new Option<FileInfo>(["--dataFile", "-d"],
                                           "Specifies the path to the input dataset file. The file must contain the input values to be used for prediction.")
                      {
                          IsRequired = true
                      });
        cmd.AddOption(new Option<FileInfo>(["--weightsFile", "-w"],
                                           "Specifies the path to the trained weights file. The weights define the state of the neural network used for prediction.")
                      {
                          IsRequired = true
                      });
        cmd.AddOption(new Option<FileInfo?>(["--outputFile", "-o"],
                                            "Optional file path to save the prediction results in plain text format."));
        cmd.AddOption(new Option<bool>("--binarizeOutput",
                                       "If enabled, predicted output values will be binarized using a fixed threshold of 0.5. Outputs >= 0.5 will be set to 1, otherwise 0."));

        cmd.Handler = CommandHandler.Create<PredictOptions, IHost>(Run);

        return cmd;
    }

    private static InputData LoadDataFile(FileInfo dataFile)
    {
        if (!dataFile.Exists)
        {
            throw new ArgumentException("The specified data file does not exist.", nameof(dataFile));
        }

        using FileStream fileStream = dataFile.OpenRead();
        using StreamReader reader = new(fileStream);

        List<double[]> inputData = new();
        string? line = reader.ReadLine();

        while (!string.IsNullOrEmpty(line))
        {
            string[] data = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

            if (data.Length == 0)
            {
                line = reader.ReadLine();

                continue;
            }

            double[] input = Array.ConvertAll(data, double.Parse);

            inputData.Add(input);

            line = reader.ReadLine();
        }

        if (inputData.Count == 0)
        {
            throw new InvalidDataException("The data file is empty or contains no valid input lines.");
        }

        double[,] sampleData = new double[inputData.Count, inputData[0].Length];
        int row = 0;

        foreach (double[] samples in inputData)
        {
            for (int col = 0; col < samples.Length; col++)
            {
                sampleData[row, col] = samples[col];
            }

            row++;
        }

        return new InputData(sampleData);
    }

    private static async Task Run(PredictOptions options, IHost host)
    {
        const string defaultHeader = "  #  | Inputs             | Raw Outputs                       ";
        const string defaultSeparator = "-----+--------------------+-----------------------------------";

        try
        {
            ConsoleOutput.PrintInfo("Loading data and weights...");

            InputData data = LoadDataFile(options.DataFile);

            Weights weights = WeightsSerializer.DeserializeFromFile(options.WeightsFile);

            BackPropagationNeuralNetwork network = new(weights);

            Console.WriteLine();
            ConsoleOutput.PrintInfo("Generating predictions...");
            Console.WriteLine();

            string header = options.BinarizeOutput ? $"{defaultHeader}| Binarized Output" : defaultHeader;
            string separator = $"{defaultSeparator}{(options.BinarizeOutput ? "+----------------------" : string.Empty)}";

            Console.WriteLine(header);
            Console.WriteLine(separator);

            for (int i = 0; i < data.Samples.GetLength(0); i++)
            {
                double[] inputs = data.Samples.GetRow(i);
                double[] outputs = network.Predict(inputs);

                string inputStr = string.Join(" ", inputs.Select(v => v.ToString("0.###")));
                string outputStr = string.Join(" ", outputs.Select(v => v.ToString("0.##0")));

                string line = $"{i + 1,4} | {inputStr,-18} | {outputStr,-33}";

                if (options.BinarizeOutput)
                {
                    string binOutputStr = string.Join(" ", outputs.Select(v => v >= 0.5 ? "1" : "0"));

                    line += $" | {binOutputStr}";
                }

                Console.WriteLine(line);
            }
        }
        catch (Exception ex)
        {
            await ConsoleOutput.PrintErrorAsync($"An error occurred during prediction: {ex.Message}");

            Environment.ExitCode = ex.HResult;
        }
    }
}