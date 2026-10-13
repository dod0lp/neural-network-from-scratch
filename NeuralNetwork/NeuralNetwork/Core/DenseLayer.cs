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
}
