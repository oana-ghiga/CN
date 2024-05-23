import numpy as np

def lu_decompose(matrix):
    n = len(matrix)
    L = np.zeros((n*(n+1))//2)
    U = np.zeros((n*(n+1))//2)

    for i in range(n):
        for j in range(i, n):
            U_index = i * (i + 1) // 2 + j - i
            U[U_index] = matrix[i][j]
            for k in range(i):
                U[U_index] -= L[i * (i + 1) // 2 + k] * U[k * (k + 1) // 2 + j - k]

        for j in range(i, n):
            if i == j:
                L[i * (i + 1) // 2 + i] = 1
            else:
                L_index = j * (j + 1) // 2 + i
                L[L_index] = matrix[j][i]
                for k in range(i):
                    L[L_index] -= L[j * (j + 1) // 2 + k] * U[k * (k + 1) // 2 + i - k]
                L[L_index] /= U[i * (i + 1) // 2 + i - i]

    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i * (i + 1) // 2 + j] * y[j]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i * (i + 1) // 2 + j - i] * x[j]
        x[i] /= U[i * (i + 1) // 2 + i - i]
    return x

def solve_with_lu(matrix, vector):
    L, U = lu_decompose(matrix)
    y = forward_substitution(L, vector)
    x = backward_substitution(U, y)
    return x

def main():
    size = int(input("Enter the size of the matrix (n): "))
    epsilon = float(input("Enter the value of epsilon: "))

    # Generate a random size x size matrix and vector
    matrix = np.random.rand(size, size)
    vector = np.random.rand(size)

    # Add epsilon to the diagonal elements to ensure the matrix is invertible
    matrix += np.eye(size) * epsilon

    solution = solve_with_lu(matrix, vector)

    print("Solution vector:")
    print(solution)

if __name__ == "__main__":
    main()
