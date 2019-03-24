import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class NN():
    def __init__(self, epochs=10, batch_size=2, layers=None, threshold=1e-3, shuffle_batches=True,
                learning_rate=1e-3):
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
        X = tf.constant(X.reshape(-1, self.input_shape).astype(np.float32))
        y = tf.constant(y.reshape(-1, self.output_shape).astype(np.float32))
        self._back_propogate(X, y)
        return self

    def predict(self, X):
        X = tf.constant(X.reshape(-1, self.input_shape).astype(np.float32))
        return self._output(X).numpy()
    
    def _output(self, X):
        output = X
        for layer in self.layers:
            output = layer._f(output @ layer.weights + layer.bias)
        return output
    
    def add_layers(self, *layers):
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
        return self.layers[0].input_shape
    
    @property
    def output_shape(self):
        return self.layers[-1].output_shape