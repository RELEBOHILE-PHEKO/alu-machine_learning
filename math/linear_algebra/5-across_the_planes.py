#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    # Check if both matrices have the same number of rows
    if len(mat1) != len(mat2):
        return None

    # Check if all rows have the same length
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # If dimensions match, perform element-wise addition
    result = []
    for row1, row2 in zip(mat1, mat2):
        row_sum = [a + b for a, b in zip(row1, row2)]  # Add corresponding elements in each row
        result.append(row_sum)

    return result

