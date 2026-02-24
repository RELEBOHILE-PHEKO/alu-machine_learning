#!/usr/bin/env python3
"Regular Markov Chain"""
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
    if not np.all(P > 0):
        if not np.all(np.linalg.matrix_power(P, n * n) > 0):
            return None
    s = np.ones((1, n)) / n
    for _ in range(1000):
        s_next = np.matmul(s, P)
        if np.allclose(s, s_next):
            return s_next
        s = s_next
    return None
