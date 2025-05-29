#!/usr/bin/env python3
"""
Calculates the inverse of a square matrix
"""


def determinant(matrix):
    """Recursive function to calculate determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(len(matrix)):
        sub = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix"""
    size = len(matrix)
    if size == 1:
        return [[1]]

    cof = []
    for i in range(size):
        row = []
        for j in range(size):
            sub = [r[:j] + r[j+1:] for idx, r in enumerate(matrix)
                   if idx != i]
            sign = (-1) ** (i + j)
            row.append(sign * determinant(sub))
        cof.append(row)
    return cof


def adjugate(matrix):
    """Returns the adjugate matrix (transpose of cofactor matrix)"""
    cof = cofactor(matrix)
    return [list(row) for row in zip(*cof)]


def inverse(matrix):
    """
    Calculates the inverse of a square matrix
    Returns None if the matrix is singular (non-invertible)
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    return [[round(el / det, 10) for el in row] for row in adj]
