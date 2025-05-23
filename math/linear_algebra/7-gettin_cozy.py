#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    # If concatenating along rows (axis 0)
    if axis == 0:
        # Make sure both matrices have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Return a new matrix with mat2 appended to mat1
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # If concatenating along columns (axis 1)
    elif axis == 1:
        # Make sure both matrices have the same number of rows
        if len(mat1) != len(mat2):
            return None
        # Append elements of mat2's rows to mat1's rows element-wise
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]

    # If axis is not 0 or 1, invalid usage
    return None

