#!/usr/bin/env python3
"""Script to create a DataFrame from a dictionary"""
import pandas as pd

# Create dictionary with the data
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create DataFrame with specified row labels
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
