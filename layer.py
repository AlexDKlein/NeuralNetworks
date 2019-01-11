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
    
    def __repr__(self):
        return f'{self.__class__.__name__}(' +\
            f'output_shape={self.output_shape}, ' +\
            f'input_shape={self.input_shape}, ' +\
            f'learning_rate={self.learning_rate})'
    
    def _construct(self, dim):
        """
        Initialize weights and set input_shape according to supplied 
        dimension. Weights are filled from a uniform distribution with 
        limits of +/- sqrt(6 / (input_shape + output_shape)).

        Parameters
        ----------
        dim: int
            New input_shape for layer.
        """
        self.input_shape=dim
        lim = np.sqrt(6 / (self.input_shape + self.output_shape))
        self.weights=np.random.uniform(-lim, lim, (self.input_shape, self.output_shape))

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
        X = self.standardize(X)
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        activated = self._activate(weighted_X)
        return activated.sum(axis=-1)
    
    def deriv(self, X):
        """
        Get the derivative of `activate` function with respect to input X.

        Parameters
        ----------
        X: ndarray of shape (-1, input_shape)
            Input arrays for the current layer.

        Returns
        ---------
        output: ndarray of shape (-1, output_shape, input_shape)
            Derivative of `activate` function for each input array in X.
        """
        X = self.standardize(X)
        weighted_X = np.einsum('di, io -> doi', X, self.weights)
        d = self._deriv(weighted_X)
        return np.einsum('doi, io -> doi', d, self.weights)
    
    def wt_deriv(self, X):
        """
        Get the derivative of `activate` function with respect to layer weights
        for input arrays in X.
        
        Parameters
        ----------
        X: ndarray of shape (-1, input_shape)
            Input arrays for the current layer.

        Returns
        ---------
        output: ndarray of shape (-1, output_shape, input_shape)
            Derivative of `activate` function wrt layer weights for each input array in X.
        """
        X = self.standardize(X)
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
        """Underlying derivative function for layer type. 
        Raises error if not overwritten in subclass.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("BaseLayer._deriv must be overwritten.")

    def standardize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
class ReLU(BaseLayer):
    activation='ReLU'
    def __init__(self, output_shape=10, input_shape=None, learning_rate=1e-3, alpha=0):
        self.alpha = alpha
        self.output_shape=output_shape
        self.learning_rate=learning_rate
        self.weights=None
        if input_shape is None:
            self.input_shape=None
        else:
            self._construct(input_shape) 

    def _activate(self, X):
        return np.where(X < 0, self.alpha * X, X)

    def _deriv(self, X):
        return np.where(X < 0, self.alpha, 1)
 
class Linear(BaseLayer):
    def _activate(self, X):
        return X

    def _deriv(self, X):
        return np.ones_like(X)

class Sigmoid(BaseLayer):
    def _activate(self, X):
        return 1 / (1 + np.exp(-X))

    def _deriv(self, X):
        f = self._activate(X)
        return f * (1 - f)

class SoftPlus(BaseLayer):
    def _activate(self, X):
        return np.log(1 + np.exp(X))

    def _deriv(self, X):
        return 1 / (1 + np.exp(-X))
    
class Sine(BaseLayer):
    def _activate(self, X):
        return np.sin(X)

    def _deriv(self, X):
        return np.cos(X)

class Sinc(BaseLayer):
    def _activate(self, X):
        return np.where(X==0, 1, np.sin(X)/X)

    def _deriv(self, X):
        return np.where(X==0, 1, (np.cos(X) / X - np.sin(X) / X**2))

class Gaussian(BaseLayer):
    def _activate(self, X):
        return np.exp(-X**2)

    def _deriv(self, X):
        f = self._activate(X)
        return -2 * f

class Ensemble(BaseLayer):
    def _construct(self, dim):
        super()._construct(dim)
        options = np.array([ReLU, Linear, Sigmoid, Gaussian, Sinc, Sine])
        idxs = np.random.choice(np.arange(len(options)), size=dim)
        self.ensemble = [l(input_shape=1) for l in options[idxs]]
        
    def _activate(self, X):
        X = X.copy()
        for i, estimator in enumerate(self.ensemble):
            X[..., i] = estimator._activate(X[..., i])
        return X

    def _deriv(self, X):
        X = X.copy()
        for i, estimator in enumerate(self.ensemble):
            X[..., i] = estimator._deriv(X[..., i])
        return X

class SoftMax(BaseLayer):
    def _activate(self, X):
        exp_X = np.exp(X)
        return exp_X / exp_X.sum(axis=0)

    def _deriv(self, X):
        pass