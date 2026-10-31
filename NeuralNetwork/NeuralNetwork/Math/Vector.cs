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

    /// <summary>Adds two equally sized vectors element by element.</summary>
    /// <param name="left">First addend.</param>
    /// <param name="right">Second addend.</param>
    /// <returns>A new vector containing <c>left[i] + right[i]</c>.</returns>
    public static Vector Add(Vector left, Vector right) {
        ValidateSameLength(left, right);

        var result = new double[left.Length];

        for (var index = 0; index < result.Length; index++) {
            result[index] = left[index] + right[index];
        }

        return new Vector(result, takeOwnership: true);
    }

    /// <summary>Subtracts one equally sized vector from another element by element.</summary>
    /// <param name="left">Vector to subtract from.</param>
    /// <param name="right">Vector to subtract.</param>
    /// <returns>A new vector containing <c>left[i] - right[i]</c>.</returns>
    public static Vector Subtract(Vector left, Vector right) {
        ValidateSameLength(left, right);

        var result = new double[left.Length];

        for (var index = 0; index < result.Length; index++) {
            result[index] = left[index] - right[index];
        }

        return new Vector(result, takeOwnership: true);
    }

    /// <summary>Multiplies every vector value by a scalar.</summary>
    /// <param name="vector">Vector to scale.</param>
    /// <param name="scalar">Value by which to multiply each element.</param>
    /// <returns>A new vector containing <c>vector[i] * scalar</c>.</returns>
    public static Vector Scale(Vector vector, double scalar) {
        var result = new double[vector.Length];

        for (var index = 0; index < result.Length; index++) {
            result[index] = vector[index] * scalar;
        }

        return new Vector(result, takeOwnership: true);
    }

    /// <summary>Multiplies two equally sized vectors element by element.</summary>
    /// <param name="left">First factor.</param>
    /// <param name="right">Second factor.</param>
    /// <returns>A new vector containing <c>left[i] * right[i]</c>.</returns>
    /// <remarks>
    /// This is the Hadamard product, not a dot product. It is used when
    /// combining backpropagation signals with activation derivatives.
    /// </remarks>
    public static Vector HadamardProduct(Vector left, Vector right) {
        ValidateSameLength(left, right);

        var result = new double[left.Length];

        for (var index = 0; index < result.Length; index++) {
            result[index] = left[index] * right[index];
        }

        return new Vector(result, takeOwnership: true);
    }

    /// <summary>Calculates the dot product of two equally sized vectors.</summary>
    /// <param name="left">First factor.</param>
    /// <param name="right">Second factor.</param>
    /// <returns>The sum of <c>left[i] * right[i]</c> for every index.</returns>
    public static double DotProduct(Vector left, Vector right) {
        ValidateSameLength(left, right);

        double total = 0;

        for (var index = 0; index < left.Length; index++) {
            total += left[index] * right[index];
        }

        return total;
    }

    /// <summary>Calculates the sum of squared differences between two equally sized vectors.</summary>
    /// <param name="left">First vector.</param>
    /// <param name="right">Second vector.</param>
    /// <returns>The value <c>sum((left[i] - right[i])^2)</c>.</returns>
    /// <remarks>
    /// This does not allocate an intermediate difference vector, making it suitable
    /// for loss calculations in a training loop.
    /// </remarks>
    public static double SquaredDistance(Vector left, Vector right) {
        ValidateSameLength(left, right);

        double total = 0;

        for (var index = 0; index < left.Length; index++) {
            var difference = left[index] - right[index];
            total += difference * difference;
        }

        return total;
    }

    /// <summary>Shorthand for <see cref="Add(Vector, Vector)"/>.</summary>
    public static Vector operator +(Vector left, Vector right) => Add(left, right);

    /// <summary>Shorthand for <see cref="Subtract(Vector, Vector)"/>.</summary>
    public static Vector operator -(Vector left, Vector right) => Subtract(left, right);

    /// <summary>Shorthand for <see cref="Scale(Vector, double)"/>.</summary>
    public static Vector operator *(Vector vector, double scalar) => Scale(vector, scalar);

    /// <summary>Shorthand for <see cref="Scale(Vector, double)"/>.</summary>
    public static Vector operator *(double scalar, Vector vector) => Scale(vector, scalar);

    /// <summary>Ensures vector operations use matching dimensions.</summary>
    private static void ValidateSameLength(Vector left, Vector right) {
        if (left.Length != right.Length) {
            throw new ArgumentException(
                $"Expected vectors with the same length, but received {left.Length} and {right.Length}.",
                nameof(right));
        }
    }
}
