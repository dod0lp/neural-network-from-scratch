# Neural Network from Scratch

Implementation of neural network in pure C#.
Rather than calling some `train()` function, I try to make it from scratch,
I will experiment with it, etc.

My initial thoguht is to make simple network for:
- odd numbers should be multiplied by `2`;
- even numbers should be divided by `2`.

## Goal

I want to implement, from my initial understanding of NN:
1. Neurons and weights.
2. Forward propagation.
3. Error calculation.
4. Backpropagation.
5. Training loop.

Fitting training data is not the same as discovering a rule
that generalizes to numbers the network has not seen.

- **Option A** (Hard): Feed it only the number and see whether it can learn the odd/even pattern.
This seems interesting, but a small standard network probably won't generalize reliably.\
- **Option B** (Easier): Give it two inputs: the number and its parity.
The network can then learn the multiplication/division behavior much more easily.

The initial network will train with data from **option B**:
1. the number
2. whether the number is odd or even.

One possible refinement is to have the network predict the multiplier (`2` or
`0.5`) and apply it to the number, instead of predicting the final result
directly.
Later I can try to do the harder approach.

## Initial path

More precise plan than written above:
1. Represent layers, neurons, weights, and biases with C# classes or arrays.
2. Implement forward propagation to produce a prediction.
3. Calculate prediction error with a loss function.
4. Implement backpropagation and gradient descent to update weights and biases.
5. Create a training loop and report how the loss changes over time.
6. Evaluate the network on unseen numbers to check generalization.
7. Optionally compare those two approaches.

All this written above is an initial direction.