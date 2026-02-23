#!/usr/bin/env python3
"""Module for finding the best number of clusters using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
            to check for (inclusive)
        kmax: positive integer containing the maximum number of clusters
            to check for (inclusive). If None, set to maximum possible
        iterations: positive integer for maximum EM iterations
        tol: non-negative float for EM tolerance
        verbose: boolean for EM verbosity

    Returns:
        best_k: best value for k based on BIC
        best_result: tuple of (pi, m, S) for best k
        l: numpy.ndarray of shape (kmax - kmin + 1) of log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1) of BIC values
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if kmax is not None and kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    k_range = kmax - kmin + 1
    l = np.zeros(k_range)
    b = np.zeros(k_range)

    best_k = None
    best_result = None
    best_bic = np.inf

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        # p = k*d*d (covariance) + k*d (means) + (k-1) (priors)
        p = k * d * d + k * d + k - 1

        l[i] = log_l
        b[i] = p * np.log(n) - 2 * log_l

        if b[i] < best_bic:
            best_bic = b[i]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
