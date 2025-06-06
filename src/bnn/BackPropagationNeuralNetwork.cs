namespace bnn;

public sealed class BackPropagationNeuralNetwork
{
    private List<List<double>> _cluster0;
    private List<List<double>> _clusterF;
    private int _hidden;
    private List<double> _hiddenLayer;
    private int _inputs;
    private int _outputs;

    public BackPropagationNeuralNetwork(int inputs = 0, int hidden = 0, int outputs = 0)
    {
        _inputs = inputs;
        _hidden = hidden;
        _outputs = outputs;

        _hiddenLayer = [];
        _cluster0 = [];
        _clusterF = [];
    }

    public bool Get(TextReader reader)
    {
        string? line = reader.ReadLine();

        if (line == null)
        {
            return false;
        }

        string[]? header = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

        if (header.Length < 3)
        {
            return false;
        }

        if (!int.TryParse(header[0], out int inputs) || !int.TryParse(header[1], out int hidden) ||
            !int.TryParse(header[2], out int outputs))
        {
            return false;
        }

        int weights = inputs + 1;

        List<List<double>> cluster0 = new(weights);

        for (int r = 0; r < hidden; r++)
        {
            List<double> neuron = ReadLineOfDoubles(reader, weights);

            if (neuron.Count == 0)
            {
                return false;
            }

            cluster0.Add(neuron);
        }

        weights = hidden + 1;

        List<List<double>> clusterF = [];

        for (int r = 0; r < outputs; r++)
        {
            List<double> neuron = ReadLineOfDoubles(reader, weights);

            if (neuron.Count == 0)
            {
                return false;
            }

            clusterF.Add(neuron);
        }

        _inputs = inputs;
        _hidden = hidden;
        _outputs = outputs;
        _cluster0 = cluster0;
        _clusterF = clusterF;

        return true;
    }

    public void Put(TextWriter writer)
    {
        writer.WriteLine($"{_inputs} {_hidden} {_outputs}");

        Show(writer);
    }

    public void Show(TextWriter writer)
    {
        int weights = _inputs + 1;

        for (int n = 0; n < _hidden; n++)
        {
            for (int w = 0; w < weights; w++)
            {
                if (w > 0)
                {
                    writer.Write('\t');
                }

                writer.Write(_cluster0[n][w]);
            }

            writer.WriteLine();
        }

        weights = _hidden + 1;

        for (int n = 0; n < _outputs; n++)
        {
            for (int w = 0; w < weights; w++)
            {
                if (w > 0)
                {
                    writer.Write('\t');
                }

                writer.Write(_clusterF[n][w]);
            }

            writer.WriteLine();
        }
    }

    private List<double> ReadLineOfDoubles(TextReader reader, int expectedCount)
    {
        string? line = reader.ReadLine();

        if (line == null)
        {
            return Enumerable.Empty<double>().ToList();
        }

        string[] tokens = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);

        if (tokens.Length != expectedCount)
        {
            return Enumerable.Empty<double>().ToList();
        }

        List<double> result = [];

        foreach (string token in tokens)
        {
            if (!double.TryParse(token, out double item))
            {
                return Enumerable.Empty<double>().ToList();
            }

            result.Add(item);
        }

        return result;
    }
}