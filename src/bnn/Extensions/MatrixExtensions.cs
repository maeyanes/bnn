namespace bnn.Extensions;

public static class MatrixExtensions
{
    public static double[] GetRow(this double[,] matrix, int rowIndex)
    {
        int cols = matrix.GetLength(1);
        double[] row = new double[cols];

        for (int c = 0; c < cols; c++)
        {
            row[c] = matrix[rowIndex, c];
        }

        return row;
    }
}