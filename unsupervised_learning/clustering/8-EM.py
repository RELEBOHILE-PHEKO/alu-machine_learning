#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k,
                             iterations=1000, tol=1e-5, verbose=False):
    """Performs expectation maximization for a GMM
    Returns: pi, m, S, g, l or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, log_like = expectation(X, pi, m, S)
    msg = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(msg.format(i, log_like.round(5)))

        pi, m, S = maximization(X, g)
        g, new_log_like = expectation(X, pi, m, S)

        if abs(log_like - new_log_like) <= tol:
            log_like = new_log_like
            break

        log_like = new_log_like

    if verbose:
        print(msg.format(i + 1, log_like.round(5)))

    return pi, m, S, g, log_like
