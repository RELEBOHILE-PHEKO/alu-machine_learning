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
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (len(matrix.shape) != 2 or
            matrix.shape[0] != matrix.shape[1] or
            matrix.size == 0):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    return "Indefinite"
