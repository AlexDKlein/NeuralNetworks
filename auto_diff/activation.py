import numpy as np
import tensorflow as tf

class Activation():
    name = None
    def __init__(self, output_shape, input_shape=None):  
        self.input_shape = input_shape
        self.output_shape = output_shape
        if input_shape is not None:
            self._construct(input_shape)
        
    def _construct(self, dim):
        self.input_shape = dim
        lim = np.sqrt(6 / (self.input_shape + self.output_shape))
        self.weights = tf.Variable(
            np.random.uniform(-lim, lim, (self.input_shape, self.output_shape)).astype(np.float32),
            trainable=True
        )
        self.bias = tf.Variable(
            np.zeros(self.output_shape).astype(np.float32),
            trainable=True
        )
        
    def activate(self, X):
        X = tf.reshape(X, (-1, self.input_shape))
        return self._f(X @ self.weights + self.bias)

def activation(func):
    class f(Activation):
        @staticmethod
        def _f(*args, **kwargs):
            return func(*args, **kwargs)
    return f

@activation
def sine(X):
    return tf.sin(X)

@activation
def relu(X):
    return tf.maximum(0.0, X)

