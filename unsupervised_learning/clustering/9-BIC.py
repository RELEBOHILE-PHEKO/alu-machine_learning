#!/usr/bin/env python3
"""This module contains a function that performs
finds the best number of clusters for a GMM using the
Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
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

    likelihoods = []
    bics = []
    best_k = None
    best_res = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        # p: free parameters = covariances + means + priors
        p = k * d * d + k * d + (k - 1)
        bic = p * np.log(n) - 2 * ll

        likelihoods.append(ll)
        bics.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_res = (pi, m, S)

    return best_k, best_res, np.array(likelihoods), np.array(bics)
