#!/usr/bin/env python3
"""
Module for transposing a NumPy matrix.
"""

import numpy as np


def np_transpose(matrix):
    """
    Transposes a NumPy matrix.

    Args:
        matrix (numpy.ndarray): The matrix to transpose.

    Returns:
        numpy.ndarray: The transposed matrix.
    """
    return np.transpose(matrix)
