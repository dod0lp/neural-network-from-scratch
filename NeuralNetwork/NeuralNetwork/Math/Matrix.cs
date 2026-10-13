namespace NeuralNetwork.Math;

/// <summary>
/// A small matrix type used for neural-network weights.
/// Rows represent output neurons and columns represent input neurons.
/// </summary>
public sealed class Matrix {
    // Backing storage; values[row, column] is the weight from one input to one output.
    private readonly double[,] values;

    /// <summary>Number of output values produced by this matrix.</summary>
    public int RowCount { get; }

    /// <summary>Number of input values required by this matrix.</summary>
    public int ColumnCount { get; }

    /// <summary>Creates a zero-filled matrix with the requested dimensions.</summary>
    public Matrix(int rowCount, int columnCount) {
        ArgumentOutOfRangeException.
            ThrowIfLessThanOrEqual(rowCount, 0, nameof(rowCount));
        ArgumentOutOfRangeException.
            ThrowIfLessThanOrEqual(columnCount, 0, nameof(columnCount));

        RowCount = rowCount;
        ColumnCount = columnCount;
        values = new double[rowCount, columnCount];
    }

    /// <summary>Gets or sets one matrix value.</summary>
    public double this[int row, int column] {
        get => values[row, column];
        set => values[row, column] = value;
    }

    /// <summary>
    /// Multiplies this matrix by a column vector and returns one value per row.
    /// </summary>
    public Vector Multiply(Vector vector) {
        if (vector.Length != ColumnCount) {
            throw new ArgumentException(
                $"Expected a vector with {ColumnCount} values, but received {vector.Length}.",
                nameof(vector));
        }

        var result = new double[RowCount];

        for (var row = 0; row < RowCount; row++) {
            // dot product of one row and input vector
            double sum = 0;

            for (var column = 0; column < ColumnCount; column++) {
                sum += values[row, column] * vector[column];
            }

            result[row] = sum;
        }

        return new Vector(result, takeOwnership: true);
    }

    /// <summary>Shorthand for <see cref="Multiply(Vector)"/>.</summary>
    public static Vector operator *(Matrix matrix, Vector vector) {
        ArgumentNullException.ThrowIfNull(matrix);

        return matrix.Multiply(vector);
    }
}
