import numpy

class Neuron():
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.weights = None
        self.threshold = None
        self.activation = None

    def _error(self):
        """
        Calculate error through neuron.
        """
        pass

    def _activate(self):
        """
        Apply activation function to input.
        """
        pass

    def _fit(self):
        pass

    def _adjust_weights(self, deriv):
        pass
        
    def _sigmoid(self):
        """
        Sigmoid activation function:
            f(x) = 1/(1 + exp(-x))
        """
        pass

    def _softmax(self):
        """
        Softmax activation function:
            f(u) = exp(u[i])/sum(exp(u[j]) for 0 < j < len(u))
        """
        pass

    def _relu(self):
        """ 
        Rectified Linear Unit activation function:
            f(x) = max(0, x)
        """
        pass

  
    
    