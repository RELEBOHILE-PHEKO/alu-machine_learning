#!/usr/bin/env python3
"""
Determines the definiteness of a matrix using eigenvalues
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix:
    - Positive definite
    - Positive semi-definite
    - Negative definite
    - Negative semi-definite
    - Indefinite
    Returns None if:
    - matrix is not a numpy.ndarray
    - matrix is not square
    - matrix is empty
    - matrix is not symmetric
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (len(matrix.shape) != 2 or
            matrix.shape[0] != matrix.shape[1] or
            matrix.size == 0):
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)  # safer for symmetric matrices

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
