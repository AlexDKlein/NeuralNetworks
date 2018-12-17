import numpy as np 
from neuron import Neuron

class Layer():
    """Basic Layer class for use in Neural Network
    """
    def __init__(self, n=1000, **kwargs):
        self.neurons = np.array([Neuron(**kwargs) for _ in range(n)])

    def _activate(self, X):
        """
        Return activation output of all neurons in layer.
        """
        return np.apply_along_axis(lambda x: x._activate(X), -1, self.neurons)
        