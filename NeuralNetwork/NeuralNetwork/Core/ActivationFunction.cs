namespace NeuralNetwork.Core;

/// <summary>
/// Holds an activation and its derivative. The derivative is used by backpropagation.
/// </summary>
/// <param name="name">Name shown when identifying the activation.</param>
/// <param name="apply">Maps a neuron's weighted sum to its output.</param>
/// <param name="derivativeFromOutput">Calculates the derivative from an activation output.</param>
public sealed class ActivationFunction(
    string name,
    Func<double, double> apply,
    Func<double, double> derivativeFromOutput) {
    /// <summary>Name of this activation.</summary>
    public string Name { get; } = name;

    /// <summary>Function applied during the forward pass.</summary>
    public Func<double, double> Apply { get; } = apply;

    /// <summary>Function used later to calculate gradients.</summary>
    public Func<double, double> DerivativeFromOutput { get; } = derivativeFromOutput;
}

/// <summary>Activation functions used by network layers.</summary>
public static class Activations {
    /// <summary>Keeps values unchanged; used by the numeric output layer.</summary>
    public static readonly ActivationFunction Linear = new(
        "Linear",
        value => value,
        _ => 1.0);

    /// <summary>Squashes hidden-layer values to the range -1 to 1.</summary>
    public static readonly ActivationFunction Tanh = new(
        "Tanh",
        System.Math.Tanh,
        output => 1.0 - (output * output));
}
