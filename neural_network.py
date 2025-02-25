import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize inputs
        self.x = np.array([0.05, 0.10])
        
        # Random weights in range [-0.5, 0.5]
        self.w = np.random.uniform(-0.5, 0.5, size=(2, 2))
        
        # Biases
        self.b = np.array([0.5, 0.7])
        
        # Target output
        self.y_true = np.array([0.8, 0.4])
    
    def tanh(self, x):
        return np.tanh(x)
    
    def mse_loss(self, y_true, y_pred):
        return 0.5 * np.sum((y_true - y_pred) ** 2)
    
    def forward_pass(self):
        z = np.dot(self.x, self.w) + self.b
        a = self.tanh(z)
        return a
    
    def compute_error(self):
        y_pred = self.forward_pass()
        return self.mse_loss(self.y_true, y_pred)
