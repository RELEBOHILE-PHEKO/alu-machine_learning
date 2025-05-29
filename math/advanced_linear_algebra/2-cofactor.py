#!/usr/bin/env python3
"""Module to calculate the cofactor matrix of a given square matrix."""


def determinant(matrix):
    """Recursively calculates the determinant of a square matrix."""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1]
                - matrix[0][1] * matrix[1][0])

    det = 0
    for c in range(n):
        sub = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub)

    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): Input matrix.

    Returns:
        list of lists: Cofactor matrix of input.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is empty or not square.
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if (len(matrix) == 0 or
            any(len(row) != len(matrix) for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cofactors = []
    for i in range(len(matrix)):
        row_cof = []
        for j in range(len(matrix)):
            # Submatrix excluding row i and column j
            sub = [row[:j] + row[j+1:]
                   for k, row in enumerate(matrix) if k != i]
            sign = (-1) ** (i + j)
            row_cof.append(sign * determinant(sub))
        cofactors.append(row_cof)

    return cofactors
