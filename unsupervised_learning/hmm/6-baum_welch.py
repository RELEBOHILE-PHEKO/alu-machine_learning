#!/usr/bin/env python3
"""Baum-Welch Algorithm for HMM"""
import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model
    Returns: Transition, Emission or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    for _ in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)
        if P is None:
            return None, None
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            num = F[:, t].reshape(-1, 1) * Transition * \
                Emission[:, Observations[t + 1]] * B[:, t + 1]
            xi[:, :, t] = num / P
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / gamma.sum(axis=1, keepdims=True)
        gamma_full = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape(-1, 1)))
        for k in range(M):
            Emission[:, k] = np.sum(
                gamma_full[:, Observations == k], axis=1)
        Emission /= gamma_full.sum(axis=1, keepdims=True)
    return Transition, Emission
