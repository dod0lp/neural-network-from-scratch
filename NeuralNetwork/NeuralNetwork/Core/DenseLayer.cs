using NeuralNetwork.Math;

namespace NeuralNetwork.Core;

/// <summary>
/// A fully connected layer. Every input connects to every output neuron.
/// </summary>
public sealed class DenseLayer {
    /// <summary>Number of values each input vector must contain.</summary>
    public int InputCount { get; }

    /// <summary>Number of values this layer produces.</summary>
    public int OutputCount { get; }

    /// <summary>Connection weights; each row belongs to one output neuron.</summary>
    public Matrix Weights { get; }

    /// <summary>One offset value added to each output neuron.</summary>
    public Vector Biases { get; }

    /// <summary>Function applied after weights and biases are combined.</summary>
    public ActivationFunction Activation { get; }

    /// <summary>Creates a fully connected layer with randomly initialized weights.</summary>
    /// <param name="inputCount">Number of values accepted by the layer.</param>
    /// <param name="outputCount">Number of neurons in the layer.</param>
    /// <param name="activation">Activation used by every neuron in this layer.</param>
    /// <param name="random">Source used to initialize the weights.</param>
    public DenseLayer(
    int inputCount,
    int outputCount,
    ActivationFunction activation,
    Random random) {

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputCount, nameof(inputCount));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputCount, nameof(outputCount));
        ArgumentNullException.ThrowIfNull(activation, nameof(activation));
        ArgumentNullException.ThrowIfNull(random, nameof(random));

        InputCount = inputCount;
        OutputCount = outputCount;
        Activation = activation;
        Weights = new Matrix(outputCount, inputCount);
        Biases = new Vector(new double[outputCount], takeOwnership: true);

        // Random values break symmetry; scale keeps initial values reasonably small.
        var scale = System.Math.Sqrt(1.0 / inputCount);

        for (var output = 0; output < outputCount; output++) {
            for (var input = 0; input < inputCount; input++) {
                Weights[output, input] = ((random.NextDouble() * 2.0) - 1.0) * scale;
            }
        }
    }

    /// <summary>
    /// Computes activation(weights * input + bias) for every output neuron.
    /// </summary>
    public Vector Forward(Vector input) {
        // Starts as weighted sums, then is replaced with activated values.
        var output = Weights * input;

        for (var neuron = 0; neuron < OutputCount; neuron++) {
            output[neuron] = Activation.Apply(output[neuron] + Biases[neuron]);
        }

        return output;
    }

    /// <summary>
    /// Propagates an output gradient through this layer and applies one gradient-descent update.
    /// </summary>
    /// <param name="input">The exact input supplied to <see cref="Forward(Vector)"/>.</param>
    /// <param name="output">The corresponding activated output from <see cref="Forward(Vector)"/>.</param>
    /// <param name="outputGradient">The derivative of loss with respect to each output value.</param>
    /// <param name="learningRate">Positive multiplier controlling the size of the parameter update.</param>
    /// <returns>The derivative of loss with respect to each input value.</returns>
    /// <remarks>
    /// The returned gradient is calculated before weights are updated, so earlier layers receive
    /// the gradient for the same parameter state that produced <paramref name="output"/>.
    /// </remarks>
    internal Vector Backpropagate(
        Vector input,
        Vector output,
        Vector outputGradient,
        double learningRate) {
        if (input.Length != InputCount) {
            throw new ArgumentException(
                $"Expected {InputCount} input values, but received {input.Length}.",
                nameof(input));
        }

        if (output.Length != OutputCount || outputGradient.Length != OutputCount) {
            throw new ArgumentException(
                $"Expected {OutputCount} output and output-gradient values.",
                nameof(output));
        }

        var deltas = new double[OutputCount];

        for (var neuron = 0; neuron < OutputCount; neuron++) {
            deltas[neuron] = outputGradient[neuron] *
                Activation.DerivativeFromOutput(output[neuron]);
        }

        var inputGradient = new double[InputCount];
        // Biases is a Vector struct exposed through a property. Keep a local copy
        // so its indexer can update the shared backing array.
        var biases = Biases;

        for (var inputIndex = 0; inputIndex < InputCount; inputIndex++) {
            double total = 0;

            for (var neuron = 0; neuron < OutputCount; neuron++) {
                total += Weights[neuron, inputIndex] * deltas[neuron];
            }

            inputGradient[inputIndex] = total;
        }

        for (var neuron = 0; neuron < OutputCount; neuron++) {
            for (var inputIndex = 0; inputIndex < InputCount; inputIndex++) {
                Weights[neuron, inputIndex] -= learningRate * deltas[neuron] * input[inputIndex];
            }

            biases[neuron] -= learningRate * deltas[neuron];
        }

        return new Vector(inputGradient, takeOwnership: true);
    }
}
