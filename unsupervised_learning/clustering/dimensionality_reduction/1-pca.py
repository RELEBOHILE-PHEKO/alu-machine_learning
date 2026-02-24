#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    Returns: T of shape (n, ndim)
    """
    A = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(A)
    W = Vt.T[:, :ndim]
    return np.matmul(A, W)
