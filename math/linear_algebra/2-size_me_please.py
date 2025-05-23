#!/usr/bin/env python3
"""
Module that contains the matrix_shape function.
This function computes the shape of a nested list (matrix).
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix (nested list).

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
              For example, a 2D matrix with 3 rows and 4 columns returns [3, 4].
    """
    shape = []  # This will store the dimensions of the matrix
    while isinstance(matrix, list):  # Keep going deeper while the current item is a list
        shape.append(len(matrix))  # Add the size of the current level (e.g., rows, columns)
        matrix = matrix[0]  # Move to the next inner list (e.g., matrix[0][0], etc.)
    return shape  # Return the list of dimensions
