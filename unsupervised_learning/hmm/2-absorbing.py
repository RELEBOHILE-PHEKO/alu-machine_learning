#!/usr/bin/env python3
"""Absorbing Markov Chain"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing
    Returns: True if absorbing, False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n:
        return False
    absorbing_states = np.where(np.diag(P) == 1)[0]
    if len(absorbing_states) == 0:
        return False
    reachable = set(absorbing_states)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i not in reachable:
                if any(P[i, j] > 0 for j in reachable):
                    reachable.add(i)
                    changed = True
    return len(reachable) == n
