import numpy as np

"""
NumPy's broadcasting allows operations on arrays of different shapes without explicitly copying or repeating data. This is not the same as actually repeating (tiling) the row or column, which would create a larger array in memory.

Why broadcasting?

Memory efficiency: Broadcasting avoids creating large intermediate arrays. For example, adding a row vector to a matrix does not require making a new matrix with the row repeated; NumPy just computes the result on the fly.
Performance: Since no extra memory is used for repeated data, operations are faster and use less RAM.
Convenience: Code is simpler and more readable.
Performance implications:

Positive: Broadcasting is much faster and more memory-efficient than manually repeating (tiling) arrays.
Negative: If you explicitly use np.tile or np.repeat to match shapes, you will use more memory and slow down your code.
Summary:
Broadcasting is a core feature for efficient, vectorized operations in NumPy. It is both faster and more memory-efficient than manual repetition.
"""

print("--- Broadcasting ---")
matrix_b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]) # Shape: (3, 3)

vector_row_for_broadcast = np.array([10, 20, 30]) # Shape: (3,)

# Add the row vector to each row of the matrix
# NumPy broadcasts the 1D vector across the rows of the 2D matrix
result_broadcast_row = matrix_b + vector_row_for_broadcast
print(f"Matrix B:\n{matrix_b}")
print(f"Vector (row-like) for Broadcasting: {vector_row_for_broadcast}")
print(f"Result of Broadcasting (Matrix B + Vector):\n{result_broadcast_row}")

# Example with a column vector (explicitly 2D)
vector_col_for_broadcast = np.array([[100], [200], [300]]) # Shape: (3, 1)
result_broadcast_col = matrix_b + vector_col_for_broadcast
print(f"\nVector (column-like) for Broadcasting:\n{vector_col_for_broadcast}")
print(f"Result of Broadcasting (Matrix B + Column Vector):\n{result_broadcast_col}")

# Scalar broadcasting (most common case)
scalar_val = 5
matrix_c = np.array([[1, 2], [3, 4]])
result_scalar_broadcast = matrix_c * scalar_val # Every element multiplied by scalar
print(f"\nMatrix C:\n{matrix_c}")
print(f"Scalar Value: {scalar_val}")
print(f"Result of Scalar Broadcasting (Matrix C * Scalar):\n{result_scalar_broadcast}")
