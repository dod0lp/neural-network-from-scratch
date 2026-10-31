using NeuralNetwork.Math;

namespace NeuralNetwork.Core;

/// <summary>
/// Loss functions used to compare a network prediction with an expected output.
/// </summary>
public static class Losses {
    /// <summary>
    /// Calculates mean squared error (MSE) between equally sized, non-empty vectors.
    /// </summary>
    /// <param name="prediction">Values produced by the network.</param>
    /// <param name="expected">Target values supplied by the training data.</param>
    /// <returns>The average of the squared difference for each output value.</returns>
    /// <remarks>
    /// MSE is zero only when every prediction exactly equals its corresponding target.
    /// It is a suitable first loss for this project's numeric output layer.
    /// </remarks>
    public static double MeanSquaredError(Vector prediction, Vector expected) {
        if (prediction.Length == 0) {
            throw new ArgumentException("Loss cannot be calculated for empty vectors.", nameof(prediction));
        }

        // SquaredDistance performs the matching-length validation without creating
        // a temporary vector of differences.
        return Vector.SquaredDistance(prediction, expected) / prediction.Length;
    }

    /// <summary>
    /// Calculates the gradient of mean squared error with respect to each prediction.
    /// </summary>
    /// <param name="prediction">Values produced by the network.</param>
    /// <param name="expected">Target values supplied by the training data.</param>
    /// <returns>
    /// A vector where each value is <c>2 * (prediction - expected) / outputCount</c>.
    /// </returns>
    /// <remarks>
    /// This gradient is the starting error signal for output-layer backpropagation.
    /// It does not yet update any weights or biases.
    /// </remarks>
    public static Vector MeanSquaredErrorGradient(Vector prediction, Vector expected) {
        ValidateComparableVectors(prediction, expected);

        var gradient = new double[prediction.Length];
        var scale = 2.0 / prediction.Length;

        for (var index = 0; index < prediction.Length; index++) {
            gradient[index] = scale * (prediction[index] - expected[index]);
        }

        return new Vector(gradient, takeOwnership: true);
    }

    /// <summary>Checks the shared preconditions of loss calculations.</summary>
    private static void ValidateComparableVectors(Vector prediction, Vector expected) {
        if (prediction.Length == 0) {
            throw new ArgumentException("Loss cannot be calculated for empty vectors.", nameof(prediction));
        }

        if (prediction.Length != expected.Length) {
            throw new ArgumentException(
                $"Expected vectors with the same length, but received {prediction.Length} and {expected.Length}.",
                nameof(expected));
        }
    }
}
