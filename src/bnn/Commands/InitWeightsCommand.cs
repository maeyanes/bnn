using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using bnn.Data;
using bnn.Options;
using bnn.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace bnn.Commands;

internal static class InitWeightsCommand
{
    public static Command Create()
    {
        Command cmd = new("init-weights", "Initialize random weights for a neuronal network");

        cmd.AddOption(new Option<int>(["--input", "-i"], "Number of input values") { IsRequired = true });
        cmd.AddOption(new Option<int>(["--hidden", "-h"], "Number of hidden layer outputs") { IsRequired = true });
        cmd.AddOption(new Option<int>(["--output", "-o"], "Number of output values") { IsRequired = true });
        cmd.AddOption(new Option<int>(["--seed", "-s"], "Optional seed for deterministic weight generation") { IsRequired = false });
        cmd.AddOption(new Option<FileInfo>("--outputFile", "Path to the output file where weights will be saved") { IsRequired = true });

        cmd.Handler = CommandHandler.Create<InitWeightsOptions, IHost>(Run);

        return cmd;
    }

    private static async Task Run(InitWeightsOptions options, IHost host)
    {
        IWeightsGeneratorService weightsGenerator = host.Services.GetRequiredService<IWeightsGeneratorService>();

        try
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Seed: {options.Seed}");
            Console.ResetColor();

            Weights weights = weightsGenerator.GenerateWeights(options);

            Console.WriteLine("Weights generated successfully");

            await using FileStream fileStream = options.OutputFile.Create();
            await using StreamWriter writer = new(fileStream);

            Console.WriteLine("Writing weights to file...");

            await writer.WriteAsync(weights.ToString());

            Console.WriteLine($"Weights saved to {options.OutputFile.FullName}");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            await Console.Error.WriteLineAsync("ERROR:");

            await Console.Error.WriteLineAsync($"An error occurred while generating weights: {ex.Message}");
            Console.ResetColor();

            Environment.ExitCode = 1;
        }
    }
}