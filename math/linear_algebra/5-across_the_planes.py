#!/usr/bin/env python3
"""
Module for performing element-wise addition of 2D matrices.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.

    Returns:
        list of lists of int/float: A new matrix representing the
        element-wise sum of mat1 and mat2, or None if matrices are
        not the same shape.
    """
    # Check if matrices have the same number of rows
    if len(mat1) != len(mat2):
        return None

    # Check if all corresponding rows have the same length
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # Perform element-wise addition
    result = []
    for row1, row2 in zip(mat1, mat2):
        row_sum = [a + b for a, b in zip(row1, row2)]
        result.append(row_sum)

    return result

