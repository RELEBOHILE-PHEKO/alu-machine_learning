import numpy as np

def np_slice(matrix: np.ndarray, axes: dict = {}) -> np.ndarray:
    """
    Slice a NumPy ndarray along specified axes.

    Parameters
    ----------
    matrix : np.ndarray
        The input array to slice.
    axes : dict, optional
        A dictionary where keys are axis indices (int), and values are tuples
        defining the slice parameters (start, stop, step) for that axis.
        If step is omitted in the tuple, it defaults to None.
        Defaults to an empty dict (no slicing).

    Returns
    -------
    np.ndarray
        A new NumPy array sliced according to the specified axes.

    Examples
    --------
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> np_slice(mat, axes={1: (1, 3)})
    array([[2, 3],
           [6, 7]])
    """
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
