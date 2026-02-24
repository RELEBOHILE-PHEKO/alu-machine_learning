#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA maintaining var fraction of variance
    Returns: W of shape (d, nd)
    """
    _, s, Vt = np.linalg.svd(X)
    cumvar = np.cumsum(s) / np.sum(s)
    nd = np.argwhere(cumvar >= var)[0, 0] + 1
    return Vt.T[:, :nd]
