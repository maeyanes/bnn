using System.Globalization;
using System.Text;
using bnn.Data;

namespace bnn.Serialization;

public static class WeightsSerializer
{
    public static Weights Deserialize(string content)
    {
        using StringReader reader = new(content);

        string? header = reader.ReadLine();

        if (header is null || !TryParseInts(header, 3, out int[] dims))
        {
            throw new InvalidDataException("Invalid header line. Expected: <input> <hidden> <output>");
        }

        int input = dims[0], hidden = dims[1], output = dims[2];

        double[,] initial = ReadMatrix(reader, hidden, input + 1);
        double[,] final = ReadMatrix(reader, output, hidden + 1);

        return new Weights(input,
                           hidden,
                           output,
                           initial,
                           final);
    }

    public static Weights DeserializeFromFile(FileInfo weightsFile)
    {
        if (!weightsFile.Exists)
        {
            throw new ArgumentException("The specified weights file does not exist.", nameof(weightsFile));
        }

        string content = File.ReadAllText(weightsFile.FullName);

        return Deserialize(content);
    }

    public static string Serialize(Weights weights)
    {
        StringBuilder builder = new();
        builder.AppendLine($"{weights.Input} {weights.Hidden} {weights.Output}");

        AppendMatrix(builder, weights.InitialCluster);
        AppendMatrix(builder, weights.FinalCluster);

        return builder.ToString();
    }

    private static void AppendMatrix(StringBuilder builder, double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (c > 0)
                {
                    builder.Append('\t');
                }

                builder.Append(matrix[r, c].ToString(CultureInfo.InvariantCulture));
            }

            builder.AppendLine();
        }
    }

    private static double[,] ReadMatrix(TextReader reader, int rows, int cols)
    {
        double[,] matrix = new double[rows, cols];

        for (int r = 0; r < rows; r++)
        {
            string? line = reader.ReadLine();

            if (line is null)
            {
                throw new InvalidDataException("Unexpected end of file while reading matrix.");
            }

            string[] parts = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length != cols)
            {
                throw new InvalidDataException($"Expected {cols} values per row, got {parts.Length}.");
            }

            for (int c = 0; c < cols; c++)
            {
                matrix[r, c] = double.Parse(parts[c], CultureInfo.InvariantCulture);
            }
        }

        return matrix;
    }

    private static bool TryParseInts(string input, int count, out int[] result)
    {
        string[] parts = input.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
        result = new int[count];

        if (parts.Length != count)
        {
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            if (!int.TryParse(parts[i], out result[i]))
            {
                return false;
            }
        }

        return true;
    }
}