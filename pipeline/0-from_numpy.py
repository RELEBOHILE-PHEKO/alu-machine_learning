#!/usr/bin/env python3
"""Function to create a DataFrame from a numpy array"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    
    Args:
        array: np.ndarray from which to create the DataFrame
        
    Returns:
        pd.DataFrame with alphabetically labeled columns (A, B, C, ...)
    """
    # Get the number of columns
    num_cols = array.shape[1]
    
    # Create column labels A, B, C, ... based on number of columns
    columns = [chr(65 + i) for i in range(num_cols)]
    
    # Create and return the DataFrame
    return pd.DataFrame(array, columns=columns)
