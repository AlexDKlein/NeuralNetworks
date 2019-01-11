import numpy as np
import matplotlib.pyplot as plt

import util

class Network():
    def __init__(self, epochs=10, batch_size=2, generations=10, layers=None, threshold=1e-3, shuffle_batches=True):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.generations = generations
        self.threshold = threshold
        self._recent_outputs = None
        self.shuffle_batches = shuffle_batches
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
    
    def __repr__(self):
        return 'Network(' + ', '.join([f'{k}={v}' for k,v in self.parameters.items()]) + ')'
    
    @property
    def parameters(self):
        return {attr: getattr(self, attr) for attr in 
        ['epochs', 'batch_size', 'generations', 'layers', 'threshold', 'shuffle_batches']}
    
    def add_layers(self, *layers):
        """Add layer objects to the network.
        New layers will be appended in order.
        
        Parameters
        ----------
        *args: layer objects
            Layers to be added to the network.
        """
        for layer in layers:
            if len(self) > 0:
                layer._construct(self[-1].output_shape) 
            self.layers.append(layer)
        
    def predict(self, X):
        """Predict the output target given input X, for the 
        current state of the network.
        Parameters
        ----------
        X: np.ndarray
            Input array with shape (n, m) where m is the input shape of layer 0.
        
        Returns
        -------
        output: np.ndarray
            Output array with shape (n,p) where p is the output shape of the final layer.

        """
        return self.output(-1, X)
    
    def activate(self, i, X):
        """Performs the activation function of layer i on input X.
        Equivalent to self.layers[i].activate(X).
        Parameters
        ---------- 
        i: int
            Index of layer from which the activation function is called.
        X: np.ndarray
            Input array with shape (n,m) where m is the input shape of layer i.

        Returns
        ----------
        output: np.ndarray
            Output array with shape (n,p) where p is the output shape of layer i.
        """
        return self[i].activate(X)
        
    def output(self, i, X):
        """Return output of layer i given input array X at layer 0.
        Parameters
        ---------- 
        X: np.ndarray
            Input array with shape (n,m) where m is the input shape of layer 0.
        Returns
        ----------
        output: np.ndarray
            Output array with shape (n,p) where p is the output shape of layer i.
        """
        while i < 0: i += len(self)
        return self[i].activate(X if i == 0 else self.output(i-1, X))

    def staged_output(self, X, start=0):
        """Return a dictionary of layer i given input array X at layer 0.
        Parameters
        ---------- 
        X: np.ndarray
            Input array with shape (n,m) where m is the input shape of layer 0.
        Returns
        ----------
        output: np.ndarray
            Output array with shape (n,p) where p is the output shape of layer i.
        """
        X_ = {start - 1: X}
        for i, layer in enumerate(self.layers[start:]):
            X_[start + i] = layer.activate(X_[start + i - 1])
        return X_

    def _deriv(self, i, X):
        return self[i]._deriv(X)
    
    def _activate(self, i, X):
        return self[i]._activate(X)
    
    def deriv(self, i, X):
        """Performs the derivative function of layer i on input X.
        Equivalent to self.layers[i].deriv(X).
        Parameters
        ---------- 
        i: int
            Layer from which the derivative function is called.
        X: np.ndarray
            Input array with shape (n,m) where m is the input shape of layer i.

        Returns
        ---------
        output: ndarray
            Array of shape (n, p, m) where m and p are the input and output shapes of layer i.
        """
        return self[i].deriv(X)
    
    def wt_deriv(self, i, X):
        """Performs the weight derivative function of layer i on input X.
        Equivalent to self.layers[i].wt_deriv(X).
        Parameters
        ---------- 
        i: int
            Layer from which the weight derivative function is called.
        X: np.ndarray
            Input array with shape (n,m) where m is the input shape of layer i.

        Returns
        ---------
        output: np.ndarray
            Array of shape (n, p, m) where m and p are the input and output shapes of layer i.
        """
        return self[i].wt_deriv(X)
        
    def weights(self, i):
        """Return the weights of layer i.
        Equivalent to self[i].weights.
        Parameters
        ---------- 
        i: int
            Layer whose weights are to be returned.
        Returns
        -------
        output: np.ndarray
            Weights array from layer i.
        """
        return self[i].weights
    
    def copy(self):
        """Return a new network with identical parameters and cloned layers.
        Returns
        -------
        output: Network
            A copy of the network.
        """
        network_copy = self.__class__(**self.parameters)
        network_copy.layers = [layer.copy() for layer in self]
        return network_copy
    
    def d(self, X, i, j=0):
        """Return derivative of output from layer i with respect to input at layer j.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        i: int
            Index of output layer
        j: int
            Index of input layer
        
        Returns
        -------
        output: np.ndarray
        """
        if i is -1:
            i = len(self) - 1
        all_outputs = self.staged_output(X)
        i_input = all_outputs[i-1] if i > 0 else X
        output = self[i].deriv(i_input)
        for k in range(i-1, j-1, -1):
            k_input = all_outputs[k-1] if k > 0 else X
            output = output @ self[k].deriv(k_input)
        return output
            
    def w(self, X, i, j=0):
        """Return the derivative of output from layer i with respect to weights at layer j.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        i: int
            Index of output layer
        j: int
            Index of input layer
        
        Returns
        -------
        output: np.ndarray
        """
        if i is -1:
            i = len(self) - 1
        
        j_input = self.output(j-1, X) if j > 0 else X
        if i == j:
            return self[j].wt_deriv(j_input)
    
        return self.d(X, i, j+1) @ self[j].wt_deriv(j_input)
             
    def error(self, X, y):
        """Return the error for the network's predictions given input X.
        Currently defaults to 1/2 MSE.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        y: np.ndarray
            Target output array
        Returns
        ---------- 
        output: float
            Total error in network output.
        """
        y_dim = len(y.shape)
        if y_dim is 1:
            y = y.reshape(-1, 1)
        return (1/2 * (self.output(-1, X) - y)**2).mean()
    
    def adj_weights(self, X, y):
        """Adjust the weights of each layer to minimize error.
        Used in 'fit' method.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        y: np.ndarray
            Target output array
        """
        for _ in range(self.generations):
            for i, layer in enumerate(self):
                for batch in util.get_batches(self.weights(i),
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle_batches):
                    eta = layer.learning_rate * self._eta(X, y, i)
                    if not np.isfinite(eta).all():
                        raise ValueError(f'Non-finite value encountered in layer {i}. Try reducing learning rate.')
                    layer.weights[batch] += eta[batch]

    def fit(self, X, y, v=False, p=False):
        """Adjust the weights of each layer to minimize error.
        Used in 'fit' method.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        y: np.ndarray
            Target output array
        Returns
        ---------
        output: self
            The fitted model
        """
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
        """Calculate the gradient of the error with respect to weights at layer i.
        Parameters
        ---------- 
        X: np.ndarray
            Input array for layer 0
        y: np.ndarray
            Target output array
        i: int
            Index of the chosen layer
        Returns
        --------
        output: np.ndarray
            Gradient of the current network error with respect to weights at layer i
        """
        err = y - self.output(-1, X)
        if i == len(self) - 1:
            return np.einsum('do, doi -> dio', err, self.w(X, i, i)).sum(axis=0)
        eta = np.einsum('do, doi -> di', err, self.d(X, len(self) - 1, i+1))
        eta = np.einsum('di, dij -> dji', eta, self.w(X,i,i))
        return eta.sum(axis=0)
    
