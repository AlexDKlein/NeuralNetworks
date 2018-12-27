import numpy as np
import matplotlib.pyplot as plt

import util

class Network():
    def __init__(self, epochs=10, batch_size=2, generations=10, layers=None, threshold=1e-3):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.generations = generations
        self.threshold = threshold
        if layers is not None:
            self.add_layers(*layers)
        
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def __call__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 2:
            return self.output(*args[::-1], **kwargs)
        elif len(args) + len(kwargs) == 3:
            pass
        return self.predict(*args, **kwargs)
    
    def add_layers(self, *layers):
        for layer in layers:
            if len(self) > 0:
                layer.construct(self[-1].output_shape) 
            self.layers.append(layer)
        
    def predict(self, X):
        return self.output(-1, X)
    
    def activate(self, i, X):
        return self[i].activate(X)
        
    def output(self, i, X):
        """Return output of layer i given input array X"""
        while i < 0: i += len(self)
        return self[i].activate(X if i == 0 else self.output(i-1, X))
 
    def _deriv(self, i, X):
        return self[i]._deriv(X)
    
    def _activate(self, i, X):
        return self[i]._activate(X)
    
    def deriv(self, i, X):
        return self[i].deriv(X)
    
    def wt_deriv(self, i, X):
        return self[i].wt_deriv(X)
        
    def weights(self, i):
        return self[i].weights
    
    def copy(self):
        network_copy = self.__class__()
        network_copy.layers = [layer.copy() for layer in self]
        return network_copy
    
    def d(self, X, i, j=0):
        """Return derivitive of output i with respect to input j.
        For some change in input, 'delta', the change in output
        can be calculated as thus:
            change in output = (d(X,-1) @ delta).sum(axis=-1) """
        if i is -1:
            i = len(self) - 1
        i_input = self.output(i-1, X) if i > 0 else X
        output = self[i].deriv(i_input)
        for k in range(i-1, j-1, -1):
            k_input = self.output(k-1, X) if k > 0 else X
            output = output @ self[k].deriv(k_input)
        return output
            
    def w(self, X, i, j=0):
        """Return the derivitive of output i with respect to weights at layer j."""
        if i is -1:
            i = len(self) - 1
        
        j_input = self.output(j-1, X) if j > 0 else X
        if i == j:
            return self[j].wt_deriv(j_input)
    
        return self.d(X, i, j+1) @ self[j].wt_deriv(j_input)
             
    def error(self, X, y):
        y_dim = len(y.shape)
        if y_dim is 1:
            y = y.reshape(-1, 1)
        return (1/2 * (self.output(-1, X) - y)**2).sum()
    
    def adj_weights(self, X, y):
        for _ in range(self.generations):
            for i, layer in enumerate(self):
                for batch in util.get_batches(self.weights(i),
                                         batch_size=self.batch_size):
                    eta = layer.learning_rate * self._eta(X, y, i)
                    if not np.isfinite(eta).all():
                        raise ValueError
                    layer.weights[batch] += eta[batch.squeeze()]

    def fit(self, X, y, v=False, p=False):
        e = []
        for i in range(self.epochs):
            self.adj_weights(X, y)
            if v: 
                print(self.error(X, y))
            if p:
                e.append(self.error(X, y))
            if self.threshold is not None:
                if self.error(X, y) < self.threshold:
                    break
        if p: 
            plt.plot(range(i +1), e)
        return self
    
    def _eta(self, X, y, i=0):
        err = y - self.output(-1, X)
        if i == len(self) - 1:
            return np.einsum('do, doi -> dio', err, self.w(X, i, i)).sum(axis=0)
        eta = np.einsum('do, doi -> di', err, self.d(X, len(self) - 1, i+1))
        eta = np.einsum('di, dij -> dji', eta, self.w(X,i,i))
        return eta.sum(axis=0)
    
