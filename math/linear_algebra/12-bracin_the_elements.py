#!/usr/bin/env python3
"""
Module for element-wise operations on numpy matrices.
Performs addition, subtraction, multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two matrices.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        tuple: A tuple of four numpy.ndarrays representing the
               element-wise sum, difference, product, and quotient.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
