import numpy as np  
from .layer import Layer

def Network():
    def __init__(self):
        self.layers = []

    def add_layer(self, layer=None, **kwargs):
        self.layers.append(layer if layer is not None else Layer(**kwargs))

    def fit(self, X, y=None):
        for layer in self.layers:
            X = layer._fit(X)
        return X

    def predict(self, X):
        for layer in self.layers:
            X = layer._activate(X)
        return X

    def _error(self, X, y):
        pass