if (args.Length > 4 || args.Length < 3)
{
    Console.WriteLine("USAGE: bnn.exe INPUTS HIDDEN OUTPUTS [seed]");

    return 0;
}

int seed = Environment.TickCount;

if (!int.TryParse(args[0], out int inputs))
{
    Console.WriteLine($"{args[0]} is not a valid integer.");

    return 1;
}

if (!int.TryParse(args[1], out int hidden))
{
    Console.WriteLine($"{args[1]} is not a valid integer.");

    return 2;
}

if (!int.TryParse(args[2], out int outputs))
{
    Console.WriteLine($"{args[2]} is not a valid integer.");

    return 3;
}


if (args.Length > 3 && !int.TryParse(args[3], out seed))
{
    Console.WriteLine($"{args[3]} is not a valid seed.");

    return 4;
}

Console.WriteLine($"Seed is {seed}");
Console.WriteLine($"{inputs} {hidden} {outputs}");

Random rnd = new(seed);

int weights = inputs + 1;

for (int r = 0; r < hidden; ++r)
{
    for (int c = 0; c < weights; ++c)
    {
        if (c > 0)
        {
            Console.Write("\t");
        }

        Console.Write($"{RandomWeight()} ");
    }

    Console.WriteLine();
}

weights = hidden + 1;

for (int r = 0; r < outputs; r++)
{
    for (int c = 0; c < weights; c++)
    {
        if (c > 0)
        {
            Console.Write("\t");
        }

        Console.Write($"{RandomWeight()} ");
    }

    Console.WriteLine();
}

return 0;


double RandomWeight()
{
    const int randMAx = 32767;
    const int bias = randMAx / 2;
    const double div = (randMAx + 1) * 10.0;

    return (rnd.Next(0, randMAx) - bias) / div;
}