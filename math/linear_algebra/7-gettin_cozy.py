#!/usr/bin/env python3
"""
Module for concatenating 2D matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Args:
        mat1 (list of lists): The first matrix.
        mat2 (list of lists): The second matrix.
        axis (int): The axis along which to concatenate
        (0 for rows, 1 for columns).

    Returns:
        list of lists: A new matrix representing the
        concatenation of mat1 and mat2,
        or None if the shapes are not compatible.
    """
    if axis == 0:
        # Ensure both matrices have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        # Ensure both matrices have the same number of rows
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]

    return None
