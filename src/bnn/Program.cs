foreach (string argument in args)
{
    if (int.TryParse(argument, out int _))
    {
        Console.WriteLine($"[I]\t{argument}");

        continue;
    }

    if (double.TryParse(argument, out double _))
    {
        Console.WriteLine($"[R]\t{argument}");

        continue;
    }

    if (char.TryParse(argument, out char _))
    {
        Console.WriteLine($"[C]\t{argument}");

        continue;
    }

    Console.WriteLine($"[S]\t{argument}");
}