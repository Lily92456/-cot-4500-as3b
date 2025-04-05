import unittest
import numpy as np

class TestMatrixOperations(unittest.TestCase):
    
    def test_problem_1(self):
        A = np.array([[2, -1, 1],
                      [1,  3, 1],
                      [-1, 5, 4]], dtype=float)
        b = np.array([6, 0, -3], dtype=float)
        expected_solution = np.linalg.solve(A, b).astype(int)
        self.assertTrue(np.array_equal(expected_solution, np.array([2, -1, 0])))

    def test_problem_2(self):
        A = np.array([[1, 1, 0, 3],
                      [2, 1, -1, 1],
                      [3, -1, -1, 2],
                      [-1, 2, 3, -1]], dtype=float)
        
        n = A.shape[0]
        L = np.eye(n)
        U = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
            for j in range(i+1, n):
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

        det_A = np.prod(np.diag(U))
        expected_det = np.linalg.det(A)
        self.assertAlmostEqual(det_A, expected_det, places=4)

    def test_problem_3(self):
        B = np.array([[9, 0, 5, 2, 1],
                      [3, 9, 1, 2, 1],
                      [0, 1, 7, 2, 3],
                      [4, 2, 3, 12, 2],
                      [3, 2, 4, 0, 8]], dtype=float)

        def diagonally_dominant(matrix):
            n = matrix.shape[0]
            for i in range(n):
                if abs(matrix[i, i]) < sum(abs(matrix[i, j]) for j in range(n) if j != i):
                    return False
            return True

        self.assertTrue(diagonally_dominant(B))

    def test_problem_4(self):
        C = np.array([[2, 2, 1],
                      [2, 3, 0],
                      [1, 0, 2]], dtype=float)
        
        def positive_definite(matrix):
            n = matrix.shape[0]
            for i in range(1, n+1):
                sub_matrix = matrix[:i, :i]  
                if np.linalg.det(sub_matrix) <= 0:
                    return False
            return True

        self.assertTrue(positive_definite(C))

if __name__ == '__main__':
    unittest.main()
