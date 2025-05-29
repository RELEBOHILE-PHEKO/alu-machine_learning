#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a given square matrix.
"""

def determinant(matrix):
    """Calculates the determinant of a square matrix."""
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        sub = [row[:col] + row[col+1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cof = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            # Get submatrix
            sub = [r[:j] + r[j+1:] for idx, r in enumerate(matrix) if idx != i]
            # Calculate determinant and apply sign
            row.append(((-1) ** (i + j)) * determinant(sub))
        cof.append(row)
    return cof


def adjugate(matrix):
    """Calculates the adjugate matrix of a square matrix."""
    cof = cofactor(matrix)
    # Transpose cofactor matrix
    adj = [list(row) for row in zip(*cof)]
    return adj
