import numpy as np

def fill_from_axis(a, axis, shape):
    """Create an array of shape 'shape' with values from input array a 
    spanning a chosen axis. Selecting axis=None will return an array with shape=(*shape, *a.shape).
    
    Examples:
    fill_from_axis(np.array([1, 2, 3]), axis=1, shape=(3, ...))
    > array([[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]])
             
    fill_from_axis(np.array([1, 2, 3]), axis=0, shape=(..., 2))
    > array([[1, 1],
             [2, 2],
             [3, 3]])
             
    fill_from_axis(np.array([1, 2, 3]), axis=None, shape=(2, 2))
    > array([[[1, 2, 3],
              [1, 2, 3]],
              
              [[1, 2, 3],
              [1, 2, 3]]])
    """
    for ax in range(len(shape)):
        if ax != axis:
            a = np.expand_dims(a, axis=ax)
            
    for ax in range(len(shape)):
        if ax != axis:
            a = np.repeat(a, shape[ax], axis=ax)
            
    return a

def get_batches(a, batch_size=1, shuffle=True, axis=None):
        """Generator yielding boolean masks for a along a given axis."""

        if axis == -1:
            axis = len(a.shape) - 1

        remaining = np.size(a, axis=axis)

        if isinstance(batch_size, float):
            size = max(int(batch_size * remaining), 1)
        elif isinstance(batch_size, int):
            size = min(batch_size, remaining)

        idx = np.arange(remaining)

        if shuffle:
            np.random.shuffle(idx)

        for n in range(0, remaining + 1, size):
            batch = (idx < n) & (n <= idx + size)
            if not np.any(batch):
                continue
            if axis is None:
                batch = batch.reshape(a.shape)
            else:
                batch = fill_from_axis(batch, axis=axis, shape=a.shape)
            yield batch