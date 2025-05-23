#!/usr/bin/env python3
"""
Module for performing matrix multiplication on numpy ndarrays.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication of two numpy ndarrays.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The product of mat1 and mat2.
    """
    # Use numpy's matmul function to multiply the matrices
    return np.matmul(mat1, mat2)
