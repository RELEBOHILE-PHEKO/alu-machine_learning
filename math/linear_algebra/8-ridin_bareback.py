#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    # Ensure matrix dimensions are valid for multiplication
    if len(mat1[0]) != len(mat2):
        return None

    # Multiply using nested list comprehensions
    result = [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
        for row in mat1
    ]
    return result

