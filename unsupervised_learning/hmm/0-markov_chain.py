#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines probability of markov chain being in a state after t iters
    Returns: numpy.ndarray of shape (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[1] != P.shape[0] or s.shape[0] != 1:
        return None
    if not isinstance(t, int) or t < 1:
        return None
    for _ in range(t):
        s = np.matmul(s, P)
    return s
