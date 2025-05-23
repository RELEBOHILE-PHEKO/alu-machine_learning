#!/usr/bin/env python3
"""
Extract specific submatrices from a 2D NumPy array using slicing.

This script demonstrates:
- Extracting the middle two rows
- Extracting the middle two columns
- Extracting a bottom-right 3x3 submatrix

No loops or conditional statements are used.
"""

import numpy as np

# Original 4x6 matrix
matrix = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24]
])

# Slice the middle two rows (rows 1 and 2)
mat1 = matrix[1:3]

# Slice the middle two columns (columns 2 and 3)
mat2 = matrix[:, 2:4]

# Slice the bottom-right 3x3 submatrix (rows 1–3, columns 3–5)
mat3 = matrix[1:, 3:]

# Display results
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))

