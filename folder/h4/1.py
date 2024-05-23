import os
import numpy as np
import re
import math
import time

def load_matrix(file_path):
    with open(file_path) as file:
        n = int(file.readline().strip())
        matrix = [{} for _ in range(n)]
        for line in file:
            parts = re.split(r"[ ,\n]+", line.strip())
            if len(parts) == 3:
                value, row, col = float(parts[0]), int(parts[1]), int(parts[2])
                if col in matrix[row]:
                    matrix[row][col] += value
                else:
                    matrix[row][col] = value
    print(f"loaded matrix from {file_path}")
    return matrix

def is_diagonally_dominant(matrix, epsilon=1e-5):
    n = len(matrix)
    main_diagonal_ok = secondary_diagonal_ok = True
    
    for i in range(n):
        main_sum = secondary_sum = 0.0
        for j, value in matrix[i].items():
            if i == j:
                main_sum += abs(value)
            if i + j == n - 1:
                secondary_sum += abs(value)
        
        if main_sum < epsilon:
            main_diagonal_ok = False
        if secondary_sum < epsilon:
            secondary_diagonal_ok = False
        
        if not main_diagonal_ok and not secondary_diagonal_ok:
            break
    
    return main_diagonal_ok, secondary_diagonal_ok

def load_vector(file_path):
    with open(file_path) as file:
        n = int(file.readline().strip())
        vector = np.array([float(file.readline().strip()) for _ in range(n)])
    return vector

def gauss_seidel(matrix, b, max_iter=1000, tol=1e-25, max_div=1e8):
    main_diagonal_ok, _ = is_diagonally_dominant(matrix, tol)
    if not main_diagonal_ok:
        print("matrix is not diagonally dominant")
        return False, None, 0

    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum_ = b[i] - sum(matrix[i].get(j, 0) * x[j] for j in range(n) if j != i)
            if i in matrix[i]:
                x[i] = sum_ / matrix[i][i]
            else:
                x[i] = sum_
        
        diff = np.linalg.norm(x - x_old)
        if diff < tol or diff > max_div:
            break
    
    if diff < tol:
        return True, x, k + 1
    else:
        return False, None, k + 1

def main():
    input_dir = "D:\\CN\\h4"
    epsilon = 1e-5
    max_iterations = 10000
    max_divergence = 1e6

    for i in range(1, 6):
        matrix_file = os.path.join(input_dir, f'a_{i}.txt')
        vector_file = os.path.join(input_dir, f'b_{i}.txt')

        A = load_matrix(matrix_file)
        b = load_vector(vector_file)

        start_time = time.time()
        success, solution, iterations = gauss_seidel(A, b, max_iterations, epsilon, max_divergence)
        duration = time.time() - start_time

        if success:
            print(f"converged in {iterations} iterations in {duration:.4f} seconds ")
            print(f"solution: {solution}")
        else:
            print(f"did not converge after {iterations} iterations in {duration:.4f} seconds")

if __name__ == "__main__":
    main()
