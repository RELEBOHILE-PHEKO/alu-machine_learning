#!/usr/bin/env python3
"""
Module that contains a function for transposing matrices.
"""


def matrix_transpose(matrix):
    """
    Transposes a 2D matrix.

    Args:
        matrix (list of lists): A 2D list representing the matrix to transpose.

    Returns:
        list of lists: A new 2D list representing the transposed matrix.
    """
    # Loop through each index i in the range of number of columns
    # For each i, create a new row by collecting matrix[row][i] from each row
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
