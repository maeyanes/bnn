using bnn;

TextReader reader;

if (Console.IsInputRedirected)
{
    reader = Console.In;
}
else
{
    if (args.Length != 1)
    {
        Console.WriteLine("USAGE: bnn.exe WEIGHTS_FILE");

        return 0;
    }

    string filename = args[0];

    if (!File.Exists(filename))
    {
        Console.WriteLine($"ERROR: File '{filename}' does not exist.");

        return 1;
    }

    reader = new StreamReader(filename);
}

BackPropagationNeuralNetwork bnn = new();

if (!bnn.Get(reader))
{
    Console.WriteLine("ERROR: Bad neural network weights data.");

    return 2;
}

bnn.Show(Console.Out);

return 0;

// double RandomWeight()
// {
//     const int randMAx = 32767;
//     const int bias = randMAx / 2;
//     const double div = (randMAx + 1) * 10.0;
//
//     return (rnd.Next(0, randMAx) - bias) / div;
// }