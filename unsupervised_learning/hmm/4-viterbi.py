#!/usr/bin/env python3
"""Viterbi Algorithm for HMM"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a HMM
    Returns: path, P or None, None on failure
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        trans_prob = V[:, t - 1].reshape(-1, 1) * Transition
        B[:, t] = np.argmax(trans_prob, axis=0)
        V[:, t] = np.max(trans_prob, axis=0) * Emission[:, Observation[t]]

    path = [int(np.argmax(V[:, T - 1]))]
    for t in range(T - 1, 0, -1):
        path.insert(0, int(B[path[0], t]))

    return path, np.max(V[:, T - 1])
