def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two matrices (nested lists) along a specified axis.

    Args:
        mat1 (list): First matrix (nested lists).
        mat2 (list): Second matrix (nested lists).
        axis (int): Axis along which to concatenate.

    Returns:
        list: New concatenated matrix or None if not possible.
    """

    def shape(matrix):
        """Get shape of nested lists."""
        if not isinstance(matrix, list):
            return ()
        if len(matrix) == 0:
            return (0,)
        return (len(matrix),) + shape(matrix[0])

    def check_shapes_compatible(s1, s2, axis):
        """Check if shapes are compatible except on concatenation axis."""
        if len(s1) != len(s2):
            # Shapes must have same number of dimensions
            return False
        for i, (d1, d2) in enumerate(zip(s1, s2)):
            if i == axis:
                continue
            if d1 != d2:
                return False
        return True

    def concat(m1, m2, axis):
        """Recursively concatenate along axis."""
        if axis == 0:
            # Concatenate lists at this level
            return m1 + m2
        else:
            # Recurse on sublists
            return [concat(sub1, sub2, axis - 1) for sub1, sub2 in zip(m1, m2)]

    shape1 = shape(mat1)
    shape2 = shape(mat2)

    if axis < 0 or axis >= len(shape1):
        # Axis out of bounds
        return None

    if not check_shapes_compatible(shape1, shape2, axis):
        return None

    # Perform concat
    return concat(mat1, mat2, axis)
