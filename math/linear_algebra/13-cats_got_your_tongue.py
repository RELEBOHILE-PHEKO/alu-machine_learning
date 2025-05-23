#!/usr/bin/env python3
"""
Module for concatenating two numpy matrices along a specified axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along the given axis.

    Args:
        mat1 (numpy.ndarray): First matrix to concatenate.
        mat2 (numpy.ndarray): Second matrix to concatenate.
        axis (int, optional): Axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: New concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
