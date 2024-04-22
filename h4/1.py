import os
import numpy as np
import re
import math
import time

def read_sparse_matrix(filename):
    matrix = None
    with open(filename) as file:
        first_line = True
        for text_line in file:
            if first_line:
                n = int(text_line)
                matrix = [dict() for i in range(n)]
                first_line = False
                continue

            fields = list(re.split("[ ,\n]+", text_line))
            fields = list(filter(lambda x: len(x) > 0, fields))

            if len(fields) != 3:
                continue

            value = float(fields[0])
            line, column = int(fields[1]), int(fields[2])

            previous_value = matrix[line].setdefault(column, 0)
            matrix[line][column] = previous_value + value

    print(f"Matricea rara din {filename} a fost incarcata cu succes folosind metoda de memorare dictionar in lista.")
    return matrix

def check_diagonal_dominance(A, eps = 0.00001):
    not_null_main_diagonal = not_null_secondary_diagonal = True

    n = len(A)

    for line in range(n):
        main_element = secondary_element = 0.0
        for column, value in A[line].items():
            if line == column:
                main_element += value
            if line + column == n - 1:
                secondary_element += value

        if abs(main_element) <= eps:
            not_null_main_diagonal = False

        if abs(secondary_element) <= eps:
            not_null_secondary_diagonal = False

        if (not not_null_main_diagonal) and (not not_null_secondary_diagonal):
            break

    return not_null_main_diagonal, not_null_secondary_diagonal

def calculate_residual_norm(A, x, b):
    Ax = [sum(row.get(j, 0) * x[j] for j in range(len(x))) for row in A]
    residual = [Ax_i - b_i for Ax_i, b_i in zip(Ax, b)]
    return max(abs(res) for res in residual)


def read_vector(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        vector = np.zeros(n).astype(float)
        for index in range(n):
                vector[index] = float(f.readline())
    return vector


def approximate_solution_using_Gauss_Seidel(A, b, max_k=1000, max_difference=1e8, eps=1e-25):
    not_null_main_diagonal, not_null_secondary_diagonal = check_diagonal_dominance(
        A, eps)

    if not not_null_main_diagonal:
        print(
            "The matrix is not diagonally dominant on the main diagonal")
        return False, "Divergence", 0

    n = b.shape[0]
    x = np.zeros(n).astype(float)
    current_k = 0
    euclidean_norm = None

    while True:
        euclidean_norm = 0.0
        for i in range(n):
            new_x_i = b[i]

            new_x_i -= sum([value * x[j] for j, value in A[i].items() if i != j])

            new_x_i /= A[i].get(i, 1.0)

            euclidean_norm += (x[i] - new_x_i) ** 2
            x[i] = new_x_i

        euclidean_norm = math.sqrt(euclidean_norm)
        current_k += 1

        if current_k >= max_k or euclidean_norm < eps or euclidean_norm > max_difference:
            break

    if euclidean_norm < eps:
        return True, x, current_k

    return False, "Divergence", current_k


def main():
    eps = 1e-5
    max_k = 10000
    max_difference = 1e6  # Define max_difference before using it
    input_dir = "D:\\CN\\h4"
    for i in range(1, 6):
        print(f"Processing files for i = {i}")
        matrix_filename = os.path.join(input_dir, f'a_{i}.txt')
        vector_filename = os.path.join(input_dir, f'b_{i}.txt')

        A = read_sparse_matrix(matrix_filename)
        b = read_vector(vector_filename)

        not_null_main_diagonal, not_null_secondary_diagonal = check_diagonal_dominance(A, eps)
        start_time = time.time()
        converged, x_approximation, iterations = approximate_solution_using_Gauss_Seidel(A, b, max_k, max_difference, eps)
        if converged:
            print(f"The method converged in {iterations} iterations.")
            print("The approximate solution is: " + str(x_approximation))
        else:
            print(f"The method did not converge after {iterations} iterations.")
        
if __name__ == "__main__":
    main()