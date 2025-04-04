import numpy as np

# Define the augmented matrix
A = np.array([[2, -1, 1],
              [1,  3, 1],
              [-1, 5, 4]], dtype=float)
b = np.array([6, 0, -3], dtype=float)

# Solve the system using NumPy's built-in solver
x = np.linalg.solve(A, b)

# Print the solution
print("Solution:", x)



# Problem 2: LU Factorization
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]], dtype=float)

# Manual LU decomposition using Doolittle's method
n = A.shape[0]
L = np.eye(n)  # Identity matrix for L
U = np.zeros((n, n))  # Zero matrix for U

for i in range(n):
    for j in range(i, n):  # Upper triangular matrix
        U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
    for j in range(i+1, n):  # Lower triangular matrix
        L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

# Determinant of A (product of diagonals of U)
det_A = np.prod(np.diag(U))

print("Determinant of A:", det_A)
print("L matrix:\n", L)
print("U matrix:\n", U)

# Problem 3: Check diagonal dominance
B = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]], dtype=float)

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        if abs(matrix[i, i]) < sum(abs(matrix[i, j]) for j in range(n) if j != i):
            return False
    return True

print("Matrix B is diagonally dominant:", is_diagonally_dominant(B))

# Problem 4: Check positive definiteness
C = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]], dtype=float)

def is_positive_definite(matrix):
    n = matrix.shape[0]
    for i in range(1, n+1):
        sub_matrix = matrix[:i, :i]  # Leading principal minors
        if np.linalg.det(sub_matrix) <= 0:
            return False
    return True

print("Matrix C is positive definite:", is_positive_definite(C))