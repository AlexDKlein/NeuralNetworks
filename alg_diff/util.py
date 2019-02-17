import numpy as np

def fill_from_axis(a, axis, shape):
    """Create an array of shape 'shape' with values from input array a 
    spanning a chosen axis. Selecting axis=None will return an array with shape=(*shape, *a.shape).
    Parameters
    -----------
    a: ndarray 
        Array to repeat over new axes.
    axis: int or None
        int - Orientation of initial values in output array.
        None - Output array will have shape (*shape, *a.shape). 
               See example below.

    shape: tuple
        Desired shape of output array. 
        Item with index = axis parameter will be ignored.
    
    Returns
    ---------
    output: ndarray of specified shape.

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
        """Generator function yielding boolean masks in the shape of 'a' 
        along a given axis.
        Parameters
        -----------
        a: ndarray
            Array on which to model masks.
        batch_size: float or int
            int - Number of True values in each mask.
            float - proportion of True values in each mask.
        shuffle: bool, default=True
            True - return batches in random order.
            False - return batches in row-first ("C") order.
        axis: int or None, default=None
            int - axis to iterate over. 
                  (All booleans sharing
                   this axis will be True
                    on the same batches).
            None - Yield a seperate mask for each index in a.
        
        Returns
        -------
        Output: generator object 

        Examples
        --------
        a = np.array([[1,2], 
                      [3,4]])

        get_batches(a, batch_size=1, shuffle=False, axis=None)
        > array([[True,  False], -> array([[False,  True]  -> ...
                 [False, False]])          [False, False]])

        get_batches(a, batch_size=1, shuffle=False, axis=0)
        > array([[True,   True], -> array([[False,  False]
                 [False, False]])         [True,   True]])
        
        get_batches(a, batch_size=2, shuffle=True, axis=None)
        > array([[True, False], -> array([[False, True]
                 [False, True]])          [True, False]])

        """

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