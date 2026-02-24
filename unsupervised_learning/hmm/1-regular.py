#!/usr/bin/env python3
"""Regular Markov Chain"""
import numpy as np


def regular(P):
    """Determines steady state probabilities of a regular markov chain
    Returns: numpy.ndarray of shape (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if not np.all(np.linalg.matrix_power(P, n * n) > 0):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady = np.real(eigenvectors[:, idx])
    steady = steady / steady.sum()
    return steady.reshape(1, n)
