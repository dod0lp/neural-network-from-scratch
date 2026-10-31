# Programmer guide

## Purpose and current scope

This is a deliberately small neural-network implementation in pure C#. It is
for learning how the mechanics work, rather than for production-scale machine
learning. It supports dense feed-forward prediction, MSE loss, backpropagation,
and stochastic-gradient-descent training.

The first experiment uses two inputs:

1. an integer;
2. a matching parity flag (`0.0` for even, `1.0` for odd).

The intended output is eventually a multiplier: `2.0` for odd inputs and
`0.5` for even inputs. The caller can then multiply the original number by the
predicted multiplier. This is a clearer learning target than directly
predicting a number whose magnitude grows with the input.

## Project layout

Unless a path is explicitly prefixed, it is relative to
`NeuralNetwork/NeuralNetwork`.

| Path | Responsibility |
| --- | --- |
| `NeuralNetwork/NeuralNetwork/Program.cs` | Small deterministic demonstration application. |
| `Core/NumberParityInput.cs` | Domain input, including validation that the parity flag agrees with the number. |
| `Core/ActivationFunction.cs` | Activation definition and the initial Linear and Tanh activations. |
| `Core/DenseLayer.cs` | Fully connected layer, parameters, initialization, and one layer forward pass. |
| `Core/FeedForwardNetwork.cs` | Layer composition, dimension validation, and public prediction API. |
| `Core/Losses.cs` | Mean-squared-error value and its derivative with respect to the prediction. |
| `Core/TrainingSample.cs` | One input/target pair for supervised learning. |
| `Math/Vector.cs` | Mutable numeric column-vector type. |
| `Math/Matrix.cs` | Matrix storage and matrix-by-vector multiplication. |

## Forward pass

`DenseLayer.Forward` computes one output for every neuron:

```text
z = W x + b
a = activation(z)
```

`x` is a column vector with `InputCount` values. `W` has one row per output
neuron and one column per input value, so its shape is
`[OutputCount, InputCount]`. `b` and `a` have `OutputCount` values.

`FeedForwardNetwork.Predict` passes the first input through each layer in
order; each layer's output becomes the next layer's input. Its constructor
rejects incompatible adjacent layer shapes before a prediction can occur.

## Parameters and initialization

Each `DenseLayer` owns mutable `Weights` and `Biases`. The initial weights are
drawn uniformly from `[-sqrt(1/inputCount), sqrt(1/inputCount)]`; biases start
at zero. The random source is supplied by the caller, so a fixed seed gives
repeatable demonstrations and future tests.

`Vector` is a `readonly struct`, but its elements can intentionally be changed
through its indexer. This lets calculations and, later, gradient descent update
values without re-allocating the vector object. Public construction copies its
input array. The internal ownership-taking constructor is only for arrays
created by the implementation and must not be used with an array that will
continue to be mutated elsewhere.

The vector API also provides `Add`, `Subtract`, `Scale`, `HadamardProduct`,
`DotProduct`, and `SquaredDistance`, with `+`, `-`, and scalar `*`.
The elementwise operations produce new vectors and require equal lengths.
`SquaredDistance` is intentionally allocation-free: it calculates the squared
error directly instead of first allocating `prediction - expected`. This is
why MSE uses it in its inner calculation.

## Loss calculation

`Losses.MeanSquaredError(prediction, expected)` implements:

```text
MSE = sum((prediction[i] - expected[i])^2) / outputCount
```

Both vectors must be non-empty and have equal lengths. The accompanying
`MeanSquaredErrorGradient` returns the derivative with respect to each
prediction:

```text
2 * (prediction[i] - expected[i]) / outputCount
```

The returned gradient starts backpropagation; it is not itself a parameter
gradient.

## Training and backpropagation

`FeedForwardNetwork.Train` caches each layer's input and activated output during
the forward pass. Starting with the MSE gradient at the final output, it then
visits layers in reverse order. `DenseLayer.Backpropagate` computes:

```text
delta = lossGradient elementwise-multiplied by activationDerivative(output)
weightGradient[row, column] = delta[row] * input[column]
biasGradient[row] = delta[row]
inputGradient = W-transpose * delta
```

The layer calculates `inputGradient` before modifying its weights. This matters:
the preceding layer must receive a gradient computed from the same weights that
produced the cached forward-pass output. It then performs the update:

```text
parameter = parameter - learningRate * parameterGradient
```

`TrainEpoch` visits samples in the supplied order, updating parameters after
each one. This is stochastic gradient descent rather than a batch update. It
returns the average loss measured immediately before each sample's update.

The next useful improvement is deterministic automated tests for a known
forward pass, vector operations, loss gradients, and a training run whose loss
falls over time. After that, add normalized inputs and mini-batch training.

## Build and run

From the repository root on .NET 10 SDK:

```powershell
dotnet build NeuralNetwork/NeuralNetwork.slnx
dotnet run --project NeuralNetwork/NeuralNetwork/NeuralNetwork.csproj
```

The current program trains on numbers 0 through 19 and predicts the multiplier
for unseen number 22. Its result is reproducible because it seeds `Random` with
`42`; it is a small learning demonstration, not a general production model.
