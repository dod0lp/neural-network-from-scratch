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

    /// <summary>
    /// Trains the network on one sample using mean squared error and stochastic gradient descent.
    /// </summary>
    /// <param name="sample">Input values and the output the network should produce.</param>
    /// <param name="learningRate">Positive size of each gradient-descent parameter update.</param>
    /// <returns>The sample's mean squared error before its parameters are updated.</returns>
    public double Train(TrainingSample sample, double learningRate) {
        return Train(sample.Input, sample.ExpectedOutput, learningRate);
    }

    /// <summary>
    /// Trains the network on one input/target pair using mean squared error and stochastic gradient descent.
    /// </summary>
    /// <param name="input">Values supplied to the first layer.</param>
    /// <param name="expectedOutput">Target values for the final layer.</param>
    /// <param name="learningRate">Positive size of each gradient-descent parameter update.</param>
    /// <returns>The sample's mean squared error before its parameters are updated.</returns>
    public double Train(Vector input, Vector expectedOutput, double learningRate) {
        ValidateTrainingArguments(input, expectedOutput, learningRate);

        // Cache each layer's input and activated output for the reverse pass.
        var layerInputs = new Vector[layers.Length];
        var layerOutputs = new Vector[layers.Length];
        var values = input;

        for (var index = 0; index < layers.Length; index++) {
            layerInputs[index] = values;
            values = layers[index].Forward(values);
            layerOutputs[index] = values;
        }

        var loss = Losses.MeanSquaredError(values, expectedOutput);
        var gradient = Losses.MeanSquaredErrorGradient(values, expectedOutput);

        for (var index = layers.Length - 1; index >= 0; index--) {
            gradient = layers[index].Backpropagate(
                layerInputs[index],
                layerOutputs[index],
                gradient,
                learningRate);
        }

        return loss;
    }

    /// <summary>
    /// Trains once on every supplied sample and returns their mean loss before each update.
    /// </summary>
    /// <param name="samples">Non-empty sequence of samples to process in its supplied order.</param>
    /// <param name="learningRate">Positive size of each gradient-descent parameter update.</param>
    /// <returns>The average pre-update mean squared error across the epoch.</returns>
    /// <remarks>
    /// Each sample updates the network immediately, so this is stochastic gradient descent,
    /// not batch gradient descent.
    /// </remarks>
    public double TrainEpoch(IEnumerable<TrainingSample> samples, double learningRate) {
        ArgumentNullException.ThrowIfNull(samples);

        double totalLoss = 0;
        var sampleCount = 0;

        foreach (var sample in samples) {
            totalLoss += Train(sample, learningRate);
            sampleCount++;
        }

        if (sampleCount == 0) {
            throw new ArgumentException("An epoch requires at least one training sample.", nameof(samples));
        }

        return totalLoss / sampleCount;
    }

    /// <summary>Checks that a single training call matches the network's dimensions and rate requirements.</summary>
    private void ValidateTrainingArguments(Vector input, Vector expectedOutput, double learningRate) {
        if (input.Length != InputCount) {
            throw new ArgumentException(
                $"Expected {InputCount} input values, but received {input.Length}.",
                nameof(input));
        }

        if (expectedOutput.Length != OutputCount) {
            throw new ArgumentException(
                $"Expected {OutputCount} target values, but received {expectedOutput.Length}.",
                nameof(expectedOutput));
        }

        if (!double.IsFinite(learningRate) || learningRate <= 0) {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "Learning rate must be a finite, positive number.");
        }
    }
}
