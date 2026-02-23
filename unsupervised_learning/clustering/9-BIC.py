#!/usr/bin/env python3
"""BIC for GMM cluster selection"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC
    Returns: best_k, best_result, l, b or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    log_likelihoods = []
    bics = []
    best_k = None
    best_res = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        # free params: k*d*(d+1)/2 covariance + k*d means + (k-1) priors
        p = k * d * (d + 1) // 2 + k * d + (k - 1)
        bic = p * np.log(n) - 2 * log_like

        log_likelihoods.append(log_like)
        bics.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_res = (pi, m, S)

    if best_k is None:
        return None, None, None, None

    return best_k, best_res, np.array(log_likelihoods), np.array(bics)
