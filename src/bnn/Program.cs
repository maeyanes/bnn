using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Hosting;
using System.CommandLine.Parsing;
using bnn.Commands;
using bnn.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

await BuildCommandLine()
      .UseHost(_ => Host.CreateDefaultBuilder(),
               host =>
               {
                   host.UseConsoleLifetime()
                       .ConfigureServices(services =>
                                          {
                                              services.Configure<ConsoleLifetimeOptions>(options => options.SuppressStatusMessages = true);

                                              services.AddTransient<IWeightsGeneratorService, WeightsGeneratorService>();
                                              services.AddTransient<INeuralNetworkTrainerService, NeuralNetworkTrainerService>();
                                          });
               })
      .UseDefaults()
      .Build()
      .InvokeAsync(args);

return;

static CommandLineBuilder BuildCommandLine()
{
    RootCommand root = new("bnn: Back Propagation Neuronal Network CLI Tools");

    root.AddCommand(InitWeightsCommand.Create());
    root.AddCommand(TrainNetworkCommand.Create());
    root.AddCommand(PredictCommand.Create());

    return new CommandLineBuilder(root);
}