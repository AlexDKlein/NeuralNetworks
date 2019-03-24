import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class NN():
    """Neural network api built over numpy and tensorflow. 
    
    Parameters
    ------------
    epochs: int, default=10
        The number of epochs to run during a fit call.
    batch_size: int, default=2
        The batch size to use during fitting.
    layers: None or List, default=None
        If a list of layers is passed, they will be passed to add_layers upon initialization.
    threshold: float, default=1e-3
        Currently unimplemented. The minimum decrease in loss to trigger the early stop criteria.
    shuffle_batches: bool, default=True
        Whether to shuffle batches used during calls to fit. 
        If False, adjustments will be made in a set order.
    learning_rate: float, default=1e-1
        Scaling factor for gradient descent. 
    """
    def __init__(self, epochs=10, batch_size=2, layers=None, threshold=1e-3, shuffle_batches=True,
                learning_rate=1e-1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers if layers is not None else []
        self.threshold = threshold
        self.shuffle_batches = shuffle_batches
        self.learning_rate = learning_rate
        self.session = None
        if layers is not None:
            self.add_layers(*layers)
    
    def fit(self, X, y=None):
        """
        Fits the model according to input X and target y.
        Parameters
        ----------
        X: array-like
            The independent variable.
        y: array-like
            The dependent variable.
        Returns
        ---------
        output: self
            The fitted model.
        """
        X = tf.constant(X.reshape(-1, self.input_shape).astype(np.float32))
        y = tf.constant(y.reshape(-1, self.output_shape).astype(np.float32))
        self._back_propogate(X, y)
        return self

    def predict(self, X):
        """
        Predict values of the target variable for input array X.
        Parameters
        ----------
        X: array-like
            The independent variable.
        Returns
        ---------
        output: np.ndarray
            The model's prediction for inputs contained in X.
        """
        X = tf.constant(X.reshape(-1, self.input_shape).astype(np.float32))
        return self._output(X).numpy()
    
    def _output(self, X):
        """
        Internal version of the predict method using tensorflow constants.
        Parameters
        ----------
        X: tf.Tensor
            The independent variable.
        Returns
        ----------
        output: tf.Tensor
            The model's prediction for inputs contained in X.
        """
        output = X
        for layer in self.layers:
            output = layer._f(output @ layer.weights + layer.bias)
        return output
    
    def add_layers(self, *layers):
        """
        Add layers to the model.
        Parameters
        ----------
        *layers: Activation layer objects
            Layers to add.
        
        """
        for layer in layers:
            if len(self.layers) > 0:
                layer._construct(self.layers[-1].output_shape) 
            self.layers.append(layer)
            
    def _back_propogate(self, X, y):
        for _ in range(self.epochs):
            for layer in self.layers:
                w,b = layer.weights, layer.bias
                self._adjust(X, y, b)
                self._adjust(X, y, w)
                
    def _adjust(self, X, y, var):
        for batch in self._get_batches(var.shape[0]):
            adj = tf.clip_by_value(
                self._neg_grad(X, y, var), -1, 1
            )
            for idx in batch:
                delta = self.learning_rate * adj[idx]
                tf.assign(var[idx], var[idx] + delta) 
            
    def _get_batches(self, n):
        n = int(n)
        batches = np.arange(n)
        if self.shuffle_batches:
            np.random.shuffle(batches)
        for i in range(n//self.batch_size):
            yield batches[i::self.batch_size]
    
    def _neg_grad(self, X, y, var):
        with tf.GradientTape() as g:
            g.watch(var)
            err = - self._error(X, y)
        return g.gradient(err, var)
    
    def _error(self, X, y):
        return ((y - self._output(X))**2)
    
    @property
    def input_shape(self):
        """The expected size of input variables."""
        return self.layers[0].input_shape
    
    @property
    def output_shape(self):
        """The expected size of target variables."""
        return self.layers[-1].output_shape