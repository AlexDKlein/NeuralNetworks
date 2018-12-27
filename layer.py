import numpy as np

class BaseLayer():
    def __init__(self, output_shape=10, input_shape=None, learning_rate=1e-3):
        self.output_shape=output_shape
        self.learning_rate=learning_rate
        self.weights=None
        if input_shape is None:
            self.input_shape=None
        else:
            self._construct(input_shape) 
        
    def __getitem__(self, key):
        return self.weights[key]
    
    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)
    
    def _construct(self, dim):
        self.input_shape=dim
        self.weights=np.random.uniform(-1, 1, (self.input_shape, self.output_shape))

    def activate(self, X):
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        activated = self._activate(weighted_X)
        return activated.sum(axis=-1)
    
    def deriv(self, X):
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        d = self._deriv(weighted_X)
        return np.einsum('doi, io -> doi', d, self.weights)
    
    def wt_deriv(self, X):
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        d = self._deriv(weighted_X)
        return np.einsum('doi, di -> doi', d, X)

    def copy(self):
        copy_layer = self.__class__(input_shape=self.input_shape, 
                                    output_shape=self.output_shape, 
                                    learning_rate=self.learning_rate)
        copy_layer.weights = self.weights.copy()
        return copy_layer

    def _activate(self, *args):
        raise NotImplementedError("BaseLayer._activate must be overwritten.")
    
    def _deriv(self, *args):
        raise NotImplementedError("BaseLayer._deriv must be overwritten.")

        
class ReLU(BaseLayer):
    def _activate(self, X):
        return np.where(X < 0, 0, X)

    def _deriv(self, X):
        return np.where(X < 0, 0, 1)
    
class Dense(BaseLayer):
    def _activate(self, X):
        return X

    def _deriv(self, X):
        return np.ones_like(X)
