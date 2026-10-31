using NeuralNetwork.Core;
using NeuralNetwork.Math;

var random = new Random(42);
var network = new FeedForwardNetwork(
    new DenseLayer(2, 4, Activations.Tanh, random),
    new DenseLayer(4, 1, Activations.Linear, random));

var samples = Enumerable.Range(0, 20)
    .Select(number => new TrainingSample(
        NumberParityInput.FromNumber(number).ToVector(),
        new Vector(number % 2 == 0 ? 0.5 : 2.0)))
    .ToArray();

double loss = 0;

for (var epoch = 0; epoch < 5_000; epoch++) {
    loss = network.TrainEpoch(samples, learningRate: 0.005);
}

// This input was not in the training set. The target is the multiplier, not
// the final transformed number: 22 * 0.5 = 11.
var input = NumberParityInput.FromNumber(22);
var predictedMultiplier = network.Predict(input)[0];

Console.WriteLine("Trained parity-multiplier demonstration");
Console.WriteLine($"Final training loss: {loss:F6}");
Console.WriteLine($"Input: [{input.Number}, {input.IsOdd}]");
Console.WriteLine($"Predicted multiplier: {predictedMultiplier:F4}");
Console.WriteLine($"Predicted result: {input.Number * predictedMultiplier:F4}");
