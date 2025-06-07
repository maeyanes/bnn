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
        Console.WriteLine($"Seed: {options.Seed}");

        IWeightsGeneratorService weightsGenerator = host.Services.GetRequiredService<IWeightsGeneratorService>();

        Weights weights = weightsGenerator.GenerateWeights(options);

        Console.WriteLine(weights.ToString());

        await Task.CompletedTask;
    }
}