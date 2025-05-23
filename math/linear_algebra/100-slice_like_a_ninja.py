import numpy as np

def np_slice(matrix: np.ndarray, axes: dict = {}) -> np.ndarray:
    """
    Slice a numpy ndarray along specified axes.

    Args:
        matrix (np.ndarray): The input matrix to slice.
        axes (dict): A dictionary where keys are axis indices (int),
                     and values are tuples representing slice parameters
                     (start, stop, step).

    Returns:
        np.ndarray: The sliced matrix.
    
    Example:
        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_slice(mat, axes={1: (1, 3)})
        array([[2, 3],
               [5, 6]])
    """
    # Prepare a slice object list for all axes
    slices = [slice(None)] * matrix.ndim
    for axis, slice_args in axes.items():
        slices[axis] = slice(*slice_args)
    return matrix[tuple(slices)]
