using NeuralNetwork.Core;
using NeuralNetwork.Math;

const int maximumNumber = 10_000;
const double numberScale = maximumNumber;
const int trainingPerParity = 4_000;
const int validationPerParity = 500;

var random = new Random(42);
var network = new FeedForwardNetwork(
    new DenseLayer(2, 4, Activations.Tanh, random),
    new DenseLayer(4, 1, Activations.Linear, random));

var splitRandom = new Random(123);
var evenNumbers = Enumerable.Range(0, maximumNumber + 1)
    .Where(number => number % 2 == 0)
    .ToList();
var oddNumbers = Enumerable.Range(0, maximumNumber + 1)
    .Where(number => number % 2 != 0)
    .ToList();

Shuffle(evenNumbers, splitRandom);
Shuffle(oddNumbers, splitRandom);

var trainingNumbers = evenNumbers.Take(trainingPerParity)
    .Concat(oddNumbers.Take(trainingPerParity))
    .ToList();
var validationNumbers = evenNumbers.Skip(trainingPerParity).Take(validationPerParity)
    .Concat(oddNumbers.Skip(trainingPerParity).Take(validationPerParity))
    .ToList();
var testNumbers = evenNumbers.Skip(trainingPerParity + validationPerParity)
    .Concat(oddNumbers.Skip(trainingPerParity + validationPerParity))
    .ToList();

// Each group is independently parity-balanced, then shuffled again so SGD does
// not receive every even sample followed by every odd sample.
Shuffle(trainingNumbers, splitRandom);
Shuffle(validationNumbers, splitRandom);
Shuffle(testNumbers, splitRandom);

TrainingSample MakeSample(int number) {
    var input = NumberParityInput.FromNumber(number);
    return new TrainingSample(
        new Vector(input.Number / numberScale, input.IsOdd ? 1.0 : 0.0),
        new Vector(input.IsOdd ? 2.0 : 0.5));
}

double EvaluateLoss(IEnumerable<int> numbers) {
    double totalLoss = 0;
    var count = 0;

    foreach (var number in numbers) {
        var sample = MakeSample(number);
        totalLoss += Losses.MeanSquaredError(network.Predict(sample.Input), sample.ExpectedOutput);
        count++;
    }

    return totalLoss / count;
}

var trainingSamples = trainingNumbers.Select(MakeSample).ToArray();

double loss = 0;

for (var epoch = 0; epoch < 100; epoch++) {
    loss = network.TrainEpoch(trainingSamples, learningRate: 0.01);
}

Console.WriteLine("Trained parity-multiplier demonstration (seeded random split)");
Console.WriteLine($"Training:   {trainingNumbers.Count} samples, final epoch loss {loss:F8}");
Console.WriteLine($"Validation: {validationNumbers.Count} samples, loss {EvaluateLoss(validationNumbers):F8}");
Console.WriteLine($"Test:       {testNumbers.Count} samples, loss {EvaluateLoss(testNumbers):F8}");

static void Shuffle<T>(IList<T> values, Random random) {
    for (var index = values.Count - 1; index > 0; index--) {
        var swapIndex = random.Next(index + 1);
        (values[index], values[swapIndex]) = (values[swapIndex], values[index]);
    }
}
