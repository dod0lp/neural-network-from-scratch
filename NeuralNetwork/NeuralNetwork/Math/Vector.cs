namespace NeuralNetwork.Math;

// vector is technically matrix, but this kinda is less error-prone with column size checks for example

/// <summary>
/// A one-dimensional column vector used for network inputs, outputs, and biases.
/// </summary>
public readonly struct Vector {
    /// <summary>Internal storage for row items.</summary>
    private readonly double[] values;

    /// <summary>Essentially number of rows.</summary>
    public int Length => values?.Length ?? 0;

    /// <summary>
    /// Creates a vector and copies the supplied values into its own storage.
    /// </summary>
    /// <param name="values">Values in column-vector order.</param>
    /// <remarks>
    /// Later changes to the source array do not affect this vector.
    /// </remarks>
    public Vector(params double[] values) {
        ArgumentNullException.ThrowIfNull(values);

        this.values = [.. values];
    }

    /// <summary>
    /// Creates a vector from an array produced by an internal calculation.
    /// </summary>
    /// <param name="values">values</param>
    /// <param name="takeOwnership">
    /// <b>true</b>: caller gives the array to Vector and must not keep mutating it afterward <br></br>
    /// <b>false</b>: it copies the array instead
    /// </param>
    internal Vector(double[] values, bool takeOwnership) {
        ArgumentNullException.ThrowIfNull(values);

        if (takeOwnership) {
            this.values = values;
        } else {
            this.values = [.. values];
        }
    }

    /// <summary>Gets or sets one value in the vector.</summary>
    public double this[int index] {
        get => values[index];
        set => values[index] = value;
    }
}