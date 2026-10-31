using NeuralNetwork.Math;

namespace NeuralNetwork.Core;

/// <summary>
/// One supervised-learning example: an input vector and the output it should produce.
/// </summary>
public readonly record struct TrainingSample {
    /// <summary>Values supplied to the network.</summary>
    public Vector Input { get; }

    /// <summary>Target values against which the network prediction is compared.</summary>
    public Vector ExpectedOutput { get; }

    /// <summary>Creates a non-empty supervised-learning example.</summary>
    /// <param name="input">Values supplied to the network.</param>
    /// <param name="expectedOutput">Target values for the supplied input.</param>
    public TrainingSample(Vector input, Vector expectedOutput) {
        if (input.Length == 0) {
            throw new ArgumentException("A training input cannot be empty.", nameof(input));
        }

        if (expectedOutput.Length == 0) {
            throw new ArgumentException("A training target cannot be empty.", nameof(expectedOutput));
        }

        Input = input;
        ExpectedOutput = expectedOutput;
    }
}
