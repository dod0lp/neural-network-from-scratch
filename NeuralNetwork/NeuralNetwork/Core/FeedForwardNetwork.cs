using NeuralNetwork.Math;

namespace NeuralNetwork.Core;

/// <summary>
/// An ordered collection of layers. Its output becomes the next layer's input.
/// </summary>
public sealed class FeedForwardNetwork {
    // Kept in forward-pass order.
    private readonly DenseLayer[] layers;

    /// <summary>Number of values required by the first layer.</summary>
    public int InputCount => layers[0].InputCount;

    /// <summary>Number of values produced by the last layer.</summary>
    public int OutputCount => layers[^1].OutputCount;

    /// <summary>Creates a network and verifies that adjacent layer sizes match.</summary>
    /// <param name="layers">Layers in forward-pass order.</param>
    public FeedForwardNetwork(params DenseLayer[] layers) {
        ArgumentNullException.ThrowIfNull(layers);

        if (layers.Length == 0) {
            throw new ArgumentException("A network must contain at least one layer.", nameof(layers));
        }

        if (layers.Any(layer => layer is null)) {
            throw new ArgumentException("A network cannot contain a null layer.", nameof(layers));
        }

        for (var index = 1; index < layers.Length; index++) {
            if (layers[index - 1].OutputCount != layers[index].InputCount) {
                throw new ArgumentException(
                    $"Layer {index - 1} outputs {layers[index - 1].OutputCount} values, " +
                    $"but layer {index} expects {layers[index].InputCount}.",
                    nameof(layers));
            }
        }

        this.layers = layers.ToArray();
    }

    /// <summary>Runs a numeric input through every layer.</summary>
    /// <param name="input">Input vector for the first layer.</param>
    /// <returns>Output vector from the final layer.</returns>
    public Vector Predict(Vector input) {
        if (input.Length != InputCount) {
            throw new ArgumentException(
                $"Expected {InputCount} input values, but received {input.Length}.",
                nameof(input));
        }

        // After each pass, values holds the previous layer's output.
        var values = input;

        foreach (var layer in layers) {
            values = layer.Forward(values);
        }

        return values;
    }

    /// <summary>
    /// Predicts from the integer-and-parity input used by this project's first
    /// experiment. It requires a network with exactly two input neurons.
    /// </summary>
    /// <param name="input">Number and matching odd/even value.</param>
    /// <returns>Output vector from the final layer.</returns>
    public Vector Predict(NumberParityInput input) {
        if (InputCount != 2) {
            throw new InvalidOperationException(
                "NumberParityInput can only be used with a network that has two input neurons.");
        }

        return Predict(input.ToVector());
    }
}
