#!/usr/bin/env python3
"""
Module for performing matrix multiplication of two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices if their dimensions are compatible.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: A new matrix which is the product
        of mat1 and mat2, or None if the matrices cannot be multiplied.
    """
    # Ensure matrix dimensions are valid for multiplication
    if len(mat1[0]) != len(mat2):
        return None

    # Perform matrix multiplication
    result = [
        [
            sum(a * b for a, b in zip(row, col))
            for col in zip(*mat2)
        ]
        for row in mat1
    ]

    return result
