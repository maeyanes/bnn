if (args.Length > 1)
{
    Console.WriteLine("Must provide just one integer.");

    return 1;
}

int n = 10;

if (args.Length == 1 && !int.TryParse(args[0], out n))
{
    Console.WriteLine($"{args[0]} is not a valid integer.");

    return 2;
}

Random rnd = new();

for (int i = 0; i < n; i++)
{
    Console.WriteLine(rnd.Next(-101, 101));
}

return 0;