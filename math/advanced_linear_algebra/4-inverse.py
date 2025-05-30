#!/usr/bin/env python3
"""
Calculates the inverse of a square matrix using its adjugate and determinant
"""

# Import the adjugate function from the previous file
adjugate = __import__('3-adjugate').adjugate


def minor(matrix, i, j):
    """
    Returns the minor matrix after removing row i and column j
    """
    return [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]


def determinant(matrix):
    """
    Recursively calculates the determinant of a matrix
    Used to determine if a matrix is invertible (det != 0)
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]
                - matrix[0][1] * matrix[1][0])

    det = 0
    for col in range(len(matrix)):
        det += ((-1) ** col) * matrix[0][col] * determinant(
            minor(matrix, 0, col))
    return det


def inverse(matrix):
    """
    Calculates the inverse of a square matrix using the formula:
    inverse = adjugate(matrix) / determinant(matrix)

    Returns None if the matrix is singular (determinant == 0)
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    return [[elem / det for elem in row] for row in adj]
