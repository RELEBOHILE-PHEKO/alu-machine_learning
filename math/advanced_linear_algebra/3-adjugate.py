#!/usr/bin/env python3
"""
Calculates the adjugate of a square matrix.
"""


def minor(matrix, i, j):
    """
    Returns the minor of a matrix by removing the i-th row and j-th column.
    """
    return [row[:j] + row[j + 1:] for idx, row in enumerate(matrix) if idx != i]


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        det += (
            (-1) ** col *
            matrix[0][col] *
            determinant(minor(matrix, 0, col))
        )
    return det


def cofactor(matrix):
    """
    Returns the cofactor matrix of a square matrix.
    """
    size = len(matrix)
    cof = []
    for i in range(size):
        row = []
        for j in range(size):
            sign = (-1) ** (i + j)
            minor_det = determinant(minor(matrix, i, j))
            row.append(sign * minor_det)
        cof.append(row)
    return cof


def transpose(matrix):
    """
    Transposes a matrix.
    """
    return [list(row) for row in zip(*matrix)]


def adjugate(matrix):
    """
    Calculates the adjugate of a square matrix.
    """
    if (
        not isinstance(matrix, list) or
        not all(isinstance(row, list) for row in matrix)
    ):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cof = cofactor(matrix)
    return transpose(cof)
