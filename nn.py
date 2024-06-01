# NEURAL NET FROM SCRATCH ༼ つ ◕_◕ ༽つ 
import numpy as np

class Dense:
    def __init__(self, inlayers, outlayers, lr):
        self.inlayers = inlayers
        self.outlayers = outlayers
        self.lr = lr
        self.weights = np.random.rand(inlayers, outlayers)
        self.bias = np.random.rand(outlayers)
    
    def forward(self, X):
        self.input = X 
        return np.dot(X, self.weights) + self.bias
    
    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = grad_output.mean(axis=0) * self.input.shape[0]

        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias

        return grad_input
