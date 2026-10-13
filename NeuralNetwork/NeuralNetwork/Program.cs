using NeuralNetwork.Core;

// This network has not been trained yet. Its weights are random, so this is
// only a check that our first building blocks can perform a forward pass.
var random = new Random(42);
var network = new FeedForwardNetwork(
    new DenseLayer(2, 4, Activations.Tanh, random),
    new DenseLayer(4, 1, Activations.Linear, random));

// The public input is an integer and a Boolean. The network converts the
// Boolean internally to 0.0 (false) or 1.0 (true).
var input = new NumberParityInput(12, false);
var prediction = network.Predict(input);

Console.WriteLine("Untrained network demonstration");
Console.WriteLine($"Input: [{input.Number}, {input.IsOdd}]");
Console.WriteLine($"Output: {prediction[0]:F4}");
