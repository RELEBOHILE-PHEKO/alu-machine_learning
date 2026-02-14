#!/usr/bin/env python3
"""This module contains a function that
calculates total intra-cluster variance for a dataset
"""
import numpy as np


def variance(X, C):
    """
    Calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset that
       will be used for K-means clustering
        - n: number of data points
        - d: number of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
       for each cluster
    
    Returns:
        var: total intra-cluster variance
        None: on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate squared distances from each point to each centroid
    # Shape: (k, n)
    squared_distances = np.sum((X - C[:, np.newaxis])**2, axis=2)

    # Find minimum squared distance for each point (to nearest centroid)
    # Shape: (n,)
    min_squared_distances = np.min(squared_distances, axis=0)

    # Sum all minimum squared distances
    var = np.sum(min_squared_distances)

    return var
