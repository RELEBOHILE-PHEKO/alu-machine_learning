#!/usr/bin/env python3
"""Baum-Welch Algorithm for HMM"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
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
        F = np.zeros((N, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = (np.dot(F[:, t - 1], Transition) *
                       Emission[:, Observations[t]])

        B = np.zeros((N, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.dot(
                Transition,
                B[:, t + 1] * Emission[:, Observations[t + 1]])

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            obs = Observations[t + 1]
            num = (F[:, t].reshape(-1, 1) * Transition *
                   Emission[:, obs] * B[:, t + 1])
            denom = np.sum(num)
            if denom == 0:
                continue
            xi[:, :, t] = num / denom

        gamma = np.sum(xi, axis=1)
        denom_t = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = np.sum(xi, axis=2) / denom_t

        gamma = np.hstack(
            (gamma,
             np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denom_e = np.sum(gamma, axis=1)
        for s in range(M):
            Emission[:, s] = np.sum(gamma[:, Observations == s], axis=1)
        Emission = np.divide(Emission, denom_e.reshape((-1, 1)))

    return Transition, Emission
