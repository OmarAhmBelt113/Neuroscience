# Description: This file is the main entry point for the program. It creates a NeuralNetwork instance, performs a forward pass, and computes the total error. The results are then printed to the console.

# Import the NeuralNetwork class from neural_network.py
from neural_network import NeuralNetwork

# Create a NeuralNetwork instance
nn = NeuralNetwork()

# Perform forward pass and compute total error
y_pred = nn.forward_pass()
total_error = nn.compute_error()


# Print results
print("Predicted Output:", y_pred)
print("Total Error:", total_error)
print("Accuracy:", 1 - total_error)