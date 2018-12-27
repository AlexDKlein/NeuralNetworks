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
        """Alternate call method for activate."""
        return self.activate(*args, **kwargs)
    
    def _construct(self, dim):
        """
        Initialize weights and set input_shape according to supplied dimension.

        Takes:
            dim: new input_shape for layer.
        """
        self.input_shape=dim
        self.weights=np.random.uniform(-1, 1, (self.input_shape, self.output_shape))

    def activate(self, X):
        """
        Apply activation function to array X.
        
        Parameters
        ----------
        X: ndarray of shape (-1, input_shape)
            Input arrays for the current layer.

        Returns
        ---------
        output: ndarray of shape (-1, output_shape)
            Output arrays of the current layer.
        """
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        activated = self._activate(weighted_X)
        return activated.sum(axis=-1)
    
    def deriv(self, X):
        """
        Get the derivitive of `activate` function with respect to input X.

        Parameters
        ----------
        X: ndarray of shape (-1, input_shape)
            Input arrays for the current layer.

        Returns
        ---------
        output: ndarray of shape (-1, output_shape, input_shape)
            Derivitive of `activate` function for each input array in X.
        """
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        d = self._deriv(weighted_X)
        return np.einsum('doi, io -> doi', d, self.weights)
    
    def wt_deriv(self, X):
        """
        Get the derivitive of `activate` function with respect to layer weights
        for input arrays in X.
        
        Parameters
        ----------
        X: ndarray of shape (-1, input_shape)
            Input arrays for the current layer.

        Returns
        ---------
        output: ndarray of shape (-1, output_shape, input_shape)
            Derivitive of `activate` function wrt layer weights for each input array in X.
        """
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        d = self._deriv(weighted_X)
        return np.einsum('doi, di -> doi', d, X)

    def copy(self):
        """
        Create a deep copy of the layer including all parameters and weights.
        
        Parameters
        ----------
        None

        Returns
        ---------
        output: A copy of the layer with identical parameters and weights. 
        """
        copy_layer = self.__class__(input_shape=self.input_shape, 
                                    output_shape=self.output_shape, 
                                    learning_rate=self.learning_rate)
        copy_layer.weights = self.weights.copy()
        return copy_layer

    def _activate(self, *args):
        """Underlying activation function for layer type. 
        Raises error if not overwritten in subclass.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("BaseLayer._activate must be overwritten.")
    
    def _deriv(self, *args):
        """Underlying derivitive function for layer type. 
        Raises error if not overwritten in subclass.
        
        Raises
        ------
        NotImplementedError
        """
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
