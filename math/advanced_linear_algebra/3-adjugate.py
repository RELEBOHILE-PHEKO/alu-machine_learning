#!/usr/bin/env python3
"""
Calculates the adjugate of a square matrix
"""


def minor(matrix, i, j):
    """
    Returns the minor of a matrix by removing the i-th row and j-th column.
    
    This is used to calculate the determinant for cofactors.
    """
    return [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.
    
    Base cases:
    - 1x1 matrix: return the single element
    - 2x2 matrix: calculate directly with ad - bc formula
    
    For larger matrices, expand along the first row using minors.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        # Calculate cofactor expansion along first row
        det += (
            (-1)**col
            * matrix[0][col]
            * determinant(minor(matrix, 0, col))
        )
    return det


def cofactor(matrix):
    """
    Builds the cofactor matrix of a square matrix.
    
    Each element is the determinant of the minor matrix times its sign,
    which depends on its position (i+j).
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
    Returns the transpose of a matrix.
    
    Rows become columns.
    """
    return [list(row) for row in zip(*matrix)]


def adjugate(matrix):
    """
    Calculates the adjugate of a square matrix.
    
    Validates input:
    - matrix must be a list of lists
    - matrix must be non-empty and square
    
    Special case:
    - 1x1 matrix adjugate is [[1]]
    
    Otherwise, returns the transpose of the cofactor matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cof = cofactor(matrix)
    return transpose(cof)
