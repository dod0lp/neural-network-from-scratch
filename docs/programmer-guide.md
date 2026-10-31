# Programmer guide

## Purpose and current scope

This is a deliberately small neural-network implementation in pure C#. It is
for learning how the mechanics work, rather than for production-scale machine
learning. The project currently supports constructing a dense feed-forward
network and making a prediction. It does not yet train its parameters.

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

It measures error only. It does not yet propagate that error through a layer
or mutate parameters.

## Next implementation slice: one trainable layer

The next focused change should make a single `DenseLayer` trainable before
generalizing backpropagation across multiple layers:

1. Retain the layer input and pre-activation/output values for a training pass.
2. Combine the loss gradient with the activation derivative to obtain the
   layer's output delta.
3. Compute `weightGradient[row, column] = delta[row] * input[column]` and
   `biasGradient[row] = delta[row]`.
4. Subtract `learningRate * gradient` from each weight and bias.
5. Add deterministic tests showing that loss decreases for a tiny sample.

Once that works, multi-layer backpropagation needs the previous layer's delta:

```text
previousDelta = (W-transpose * currentDelta) elementwise-multiplied by activationDerivative
```

Keep the weight orientation described above; it makes the forward pass and
this transpose relationship unambiguous.

## Build and run

From the repository root on .NET 10 SDK:

```powershell
dotnet build NeuralNetwork/NeuralNetwork.slnx
dotnet run --project NeuralNetwork/NeuralNetwork/NeuralNetwork.csproj
```

The current program intentionally prints an untrained prediction. Its numeric
result is reproducible because it seeds `Random` with `42`, but it has no
learning significance until training is added.
