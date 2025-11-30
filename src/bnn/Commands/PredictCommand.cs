using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using System.Text;
using bnn.Activation;
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

        Option<string> activationOption = new(["--activation", "-a"],
                                              "Activation function to use: sigmoid, relu, tanh, sigmoidPlus or tanhPlus (default: sigmoid)")
                                          {
                                              IsRequired = false
                                          };

        activationOption.SetDefaultValue("sigmoid");
        activationOption.AddValidator(result =>
                                      {
                                          string? value = result.GetValueOrDefault<string>()?.ToLowerInvariant();

                                          if (value is not ("sigmoid" or "relu" or "tanh" or "sigmoidplus" or "tanhplus"))
                                          {
                                              result.ErrorMessage =
                                                  "Activation must be one of: sigmoid, relu, tanh, sigmoidPlus, tanhPlus.";
                                          }
                                      });

        cmd.AddOption(activationOption);

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
        try
        {
            ConsoleOutput.PrintInfo("Loading data and weights...");

            InputData data = LoadDataFile(options.DataFile);

            Weights weights = WeightsSerializer.DeserializeFromFile(options.WeightsFile);

            BackPropagationNeuralNetwork network = new(weights);

            (Func<double, double> activation, Func<double, double> activationDerivative) = options.Activation.ToLowerInvariant() switch
                                                                                           {
                                                                                               "sigmoid" => ActivationFunctions.Sigmoid,
                                                                                               "relu" => ActivationFunctions.ReLu,
                                                                                               "tanh" => ActivationFunctions.Tanh,
                                                                                               "sigmoidplus" => ActivationFunctions
                                                                                                   .SigmoidPlus,
                                                                                               "tanhplus" => ActivationFunctions.TanhPlus,
                                                                                               { } unknown =>
                                                                                                   throw new
                                                                                                       ArgumentException($"Unsupported activation function: {unknown}")
                                                                                           };

            network.SetActivationFunction(activation, activationDerivative);

            Console.WriteLine();
            ConsoleOutput.PrintInfo($"Generating predictions with {options.Activation} activation...");
            Console.WriteLine();

            bool saveOutputs = options.OutputFile is not null;

            List<string[]> rawOutputs = new();
            List<string[]> binarizedOutputs = new();
            List<string[]> inputStrings = new();
            StringBuilder outputBuilder = new();

            for (int i = 0; i < data.Samples.GetLength(0); i++)
            {
                double[] inputs = data.Samples.GetRow(i);
                double[] outputs = network.Predict(inputs);

                inputStrings.Add(inputs.Select(v => v.ToString("0.##0")).ToArray());
                rawOutputs.Add(outputs.Select(v => v.ToString("0.##0")).ToArray());

                if (options.BinarizeOutput)
                {
                    binarizedOutputs.Add(outputs.Select(v => v >= 0.5 ? "1" : "0").ToArray());

                    outputs = outputs.Select(v => v >= 0.5 ? 1.0 : 0.0).ToArray();
                }

                if (!saveOutputs)
                {
                    continue;
                }

                outputBuilder.AppendJoin(" ", outputs);
                outputBuilder.AppendLine();
            }

            // Calcular anchos máximos por sección
            int inputWidth = inputStrings.Max(arr => string.Join(" ", arr).Length);
            int outputWidth = rawOutputs.Max(arr => string.Join(" ", arr).Length);
            int binarizedWidth = options.BinarizeOutput ? binarizedOutputs.Max(arr => string.Join(" ", arr).Length) : 0;

            // Crear encabezados dinámicamente
            string header = $"  #  | {"Inputs".PadRight(inputWidth)} | {"Raw Outputs".PadRight(outputWidth)}";
            string separator = $"-----+{new string('-', inputWidth + 2)}+{new string('-', outputWidth + 2)}";

            if (options.BinarizeOutput)
            {
                header += $" | {"Binarized Output".PadRight(binarizedWidth)}";
                separator += "+-----------------";
            }

            Console.WriteLine(header);
            Console.WriteLine(separator);

            // Imprimir resultados
            for (int i = 0; i < inputStrings.Count; i++)
            {
                int[] perOutputWidths = Enumerable.Range(0, inputStrings[0].Length)
                                                  .Select(col => inputStrings.Max(row => row[col].Length))
                                                  .ToArray();

                string inputStr = string.Join(" ", inputStrings[i].Select((val, idx) => val.PadLeft(perOutputWidths[idx])));

                perOutputWidths = Enumerable.Range(0, rawOutputs[0].Length).Select(col => rawOutputs.Max(row => row[col].Length)).ToArray();

                string outputStr = string.Join(" ", rawOutputs[i].Select((val, idx) => val.PadLeft(perOutputWidths[idx])));

                string line = $"{i + 1,4} | {inputStr.PadRight(inputWidth)} | {outputStr}";

                if (options.BinarizeOutput)
                {
                    string binStr = string.Join(" ", binarizedOutputs[i]);
                    line += $" | {binStr.PadRight(binarizedWidth)}";
                }

                Console.WriteLine(line);
            }

            if (saveOutputs)
            {
                await using FileStream fileStream = options.OutputFile!.Create();
                await using StreamWriter textWriter = new(fileStream);

                await textWriter.WriteLineAsync(outputBuilder.ToString().Trim());
            }
        }
        catch (Exception ex)
        {
            await ConsoleOutput.PrintErrorAsync($"An error occurred during prediction: {ex.Message}");

            Environment.ExitCode = ex.HResult;
        }
    }
}