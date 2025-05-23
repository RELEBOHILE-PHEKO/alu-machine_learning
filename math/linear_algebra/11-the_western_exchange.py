#!/usr/bin/env python3
"""
Module for transposing a NumPy matrix without using imports.
"""


def np_transpose(matrix):
    """
    Transposes a given NumPy ndarray.

    Args:
        matrix (numpy.ndarray): The matrix to be transposed.

    Returns:
        numpy.ndarray: A new transposed matrix.
    """
    return matrix.transpose()
