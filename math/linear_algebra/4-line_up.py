#!/usr/bin/env python3
"""
Module that contains a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays (lists) element-wise.

    Args:
        arr1 (list): The first list of integers/floats.
        arr2 (list): The second list of integers/floats.

    Returns:
        list or None: A list containing the sum of corresponding elements,
                      or None if the input lists are not the same length.
    """
    if len(arr1) != len(arr2):
        return None  # Return None if array lengths don't match

    return [a + b for a, b in zip(arr1, arr2)]  # Element-wise addition
