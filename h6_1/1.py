import os
import numpy as np
import time

def read_sparse_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        A = []
        for i in range(n):
            row_data = list(map(float, f.readline().split(',')))
            A.append({j: value for j, value in enumerate(row_data) if value != 0})
            if len(A[i]) == 0 or i not in A[i]:
                raise ValueError(f"Diagonal element at row {i} is zero.")
    return A

def read_vector(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        vector = np.zeros(n)
        for i, line in enumerate(f):
            if line.strip():
                value = float(line)
                vector[i] = value
    return vector

def gauss_seidel_sparse(A, b, x=None, eps=1e-6, max_iterations=10000):
    n = len(b)
    if x is None:
        x = np.zeros(n)  # Initial guess
    
    for _ in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            row = A[i]
            Aii = row.get(i, 0)
            if Aii == 0:
                raise ValueError(f"Zero diagonal element at row {i}")
            Aix = sum(value * x[j] for j, value in row.items() if j != i)
            x[i] = (b[i] - Aix) / Aii
        if np.max(np.abs(x - x_old)) < eps:
            break
    
    return x

def calculate_residual_norm(A, x, b):
    Ax = np.zeros_like(b)
    for i, row in enumerate(A):
        for j, value in row.items():
            Ax[i] += value * x[j]
    residual = Ax - b
    return np.max(np.abs(residual))

def main():
    input_dir = "D:\\CN\\h4"
    for i in range(1, 6):
        print(f"Processing files for i = {i}")
        matrix_filename = os.path.join(input_dir, f'a_{i}.txt')
        vector_filename = os.path.join(input_dir, f'b_{i}.txt')

        A = read_sparse_matrix(matrix_filename)
        vector = read_vector(vector_filename)
        start_time = time.time()
        try:
            x = gauss_seidel_sparse(A, vector, eps=1e-6, max_iterations=10000)
            residual_norm = calculate_residual_norm(A, x, vector)
            print(f"The norm of the difference between Ax and b is: {residual_norm}")
            print(f"For the system in {matrix_filename} and {vector_filename}:")
            print("The approximate solution is:", x)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        except ValueError as e:
            print(f"Error for the system in {matrix_filename} and {vector_filename}: {e}")

if __name__ == "__main__":
    main()
