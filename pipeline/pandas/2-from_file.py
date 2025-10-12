#!/usr/bin/env python3
"""Function to load data from a file as a DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame
    
    Args:
        filename: the file to load from
        delimiter: the column separator
        
    Returns:
        pd.DataFrame with the loaded data
    """
    return pd.read_csv(filename, delimiter=delimiter)
