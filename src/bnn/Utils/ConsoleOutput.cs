namespace bnn.Utils;

public static class ConsoleOutput
{
    public static void PrintError(string message)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.Error.WriteLine(message);
        Console.ResetColor();
    }

    public static async Task PrintErrorAsync(string message, bool useLabel = true)
    {
        Console.ForegroundColor = ConsoleColor.Red;

        if (useLabel)
        {
            await Console.Error.WriteAsync("[ERROR] ");
        }

        await Console.Error.WriteLineAsync(message);
        Console.ResetColor();
    }

    public static void PrintInfo(string message)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(message);
        Console.ResetColor();
    }

    public static void PrintSuccess(string message)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine(message);
        Console.ResetColor();
    }

    public static void PrintWarning(string message)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(message);
        Console.ResetColor();
    }
}