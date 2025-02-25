# Simple Neural Network Implementation

## Description

This repository contains a basic neural network implementation from scratch using NumPy. The network uses the hyperbolic tangent (tanh) activation function and the mean squared error (MSE) loss function. The code provides a simple example of a forward pass and error calculation.

**Note:** This implementation is a simplified example and does not include backpropagation for learning.

## Files

* `neural_network.py`: Contains the `NeuralNetwork` class, which defines the network architecture, activation function, loss function, and forward pass.
* `main.py`: The main script that creates a `NeuralNetwork` instance, performs a forward pass, computes the error, and prints the results.

## Dependencies

* `numpy`: For numerical computations.

## Usage

1.  **Clone the repository (optional):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Run the `main.py` script:**

    ```bash
    python main.py
    ```

    This will execute the forward pass and print the predicted output, total error, and an "accuracy" value (1-error).

## Explanation

### `neural_network.py`

* The `NeuralNetwork` class initializes the network with:
    * Input values (`x`).
    * Random weights (`w`).
    * Biases (`b`).
    * Target output (`y_true`).
* The `tanh()` method implements the hyperbolic tangent activation function.
* The `mse_loss()` function calculates the mean squared error between the predicted and target outputs.
* The `forward_pass()` method performs the forward pass calculation:
    * Calculates the weighted sum of inputs and biases (`z`).
    * Applies the tanh activation function to `z` (`a`).
    * It currently has a return of nothing, and should return a.
* The `compute_error()` method calculates the MSE loss.

### `main.py`

* Creates an instance of the `NeuralNetwork` class.
* Calls the `forward_pass()` and `compute_error()` methods.
* Prints the predicted output, total error, and "accuracy" to the console.

## Notes

* This is a very basic implementation and lacks backpropagation for training the network.
* The "accuracy" value (1 - error) provided is a very basic estimate and is not a true representation of accuracy in a trained network.
* The `forward_pass()` function in `neural_network.py` currently returns `None`. To get the predicted output, it should return the activated output `a`.
* The weights and biases are initialized randomly, so the results will vary each time the script is run.
* This example uses a fixed input and target output for demonstration purposes. To use this network for a real-world problem, you would need to modify the code to load and process your own data.
* The network has a fixed architecture (2 input neurons, 2 output neurons). To create a more complex network, you would need to add more layers and neurons.
* To create a functional neural network, backpropagation needs to be implemented.
