if (args.Length > 2)
{
    Console.WriteLine("USAGE: bnn.exe [n] [seed]");

    return 0;
}

int n = 10;
int seed = Environment.TickCount;

if (args.Length > 1 && !int.TryParse(args[0], out n))
{
    Console.WriteLine($"{args[0]} is not a valid integer.");

    return 1;
}

if (args.Length == 2 && !int.TryParse(args[1], out seed))
{
    Console.WriteLine($"{args[1]} is not a valid seed.");

    return 1;
}

Console.WriteLine($"Seed is {seed}");

Random rnd = new(seed);

for (int i = 0; i < n; i++)
{
    Console.WriteLine(RandomWeight());
}

return 0;


double RandomWeight()
{
    const int randMAx = 32767;
    const int bias = randMAx / 2;
    const double div = (randMAx + 1) * 10.0;

    return (rnd.Next(0, randMAx) - bias) / div;
}