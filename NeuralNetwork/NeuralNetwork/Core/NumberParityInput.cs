using NeuralNetwork.Math;

namespace NeuralNetwork.Core;

/// <summary>
/// Represents a whole number and its parity for the first experiment.
/// </summary>
public readonly record struct NumberParityInput {
    /// <summary>Number supplied to the experiment.</summary>
    public int Number { get; }

    /// <summary><b>true</b> when <see cref="Number"/> is odd.</summary>
    public bool IsOdd { get; }

    /// <summary>Creates input and rejects a parity value that does not match the number.</summary>
    /// <param name="number">Number supplied to the experiment.</param>
    /// <param name="isOdd">Whether <paramref name="number"/> is odd.</param>
    public NumberParityInput(int number, bool isOdd) {
        if ((number % 2 != 0) != isOdd) {
            throw new ArgumentException(
                $"The supplied parity does not match {number}.",
                nameof(isOdd));
        }

        Number = number;
        IsOdd = isOdd;
    }

    /// <summary>
    /// Converts input to the network's form [number, isOdd].
    /// </summary>
    /// <returns>A numeric vector containing the number and parity flag.</returns>
    public Vector ToVector() {
        if (IsOdd) {
            return new Vector(Number, 1);
        }

        return new Vector(Number, 0);
    }

    /// <summary>Builds input and calculates its parity from the supplied number.</summary>
    /// <param name="number">Number supplied to the experiment.</param>
    public static NumberParityInput FromNumber(int number) {
        return new(number, number % 2 != 0);
    }
}
