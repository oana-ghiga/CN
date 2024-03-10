import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U


def determinant_LU(L, U):
    return np.prod(np.diag(L)) * np.prod(np.diag(U))


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def check_solution(A_init, b_init, x_LU):
    norm = np.linalg.norm(A_init @ x_LU - b_init, ord=2)
    return norm


def main():
    # Input data
    n = int(input("Enter the size of the system (n): "))
    epsilon = float(input("Enter the precision of the calculations (epsilon): "))

    # Generate random matrix A and vector b for testing
    A_init = np.random.rand(n, n)
    b_init = np.random.rand(n)

    # LU decomposition
    L, U = lu_decomposition(A_init)

    # Determinant of A
    det_A = determinant_LU(L, U)
    ####1. Calculate LU decomposition####

    # Solve the system using LU decomposition
    y = forward_substitution(L, b_init)
    x_LU = backward_substitution(U, y)
    ####3. Using LU decomposition, calculate the solution x_LU####

    # Check the solution
    norm_check = check_solution(A_init, b_init, x_LU)
    print(f"Norm check: {norm_check}")
    ####4. Check the calculated solution####

    # Display solution and inverse
    x_lib = np.linalg.solve(A_init, b_init)
    A_inv_lib = np.linalg.inv(A_init)

    # Display results
    print("\nResults:")
    print("LU Decomposition:")
    print("L matrix:")
    print(L)
    print("\nU matrix:")
    print(U)
    print("\nDeterminant of A:", det_A)

    print("\nSolution using LU decomposition (x_LU):", x_LU)
    ####2. Display the solution using LU decomposition####

    print("\nSolution using NumPy library (x_lib):", x_lib)

    # Check the solutions
    norm_x_diff = np.linalg.norm(x_LU - x_lib, ord=2)
    norm_xLU_Ainvlib_diff = np.linalg.norm(x_LU - A_inv_lib @ b_init, ord=2)

    print("\nSolution Comparison:")
    print("||x_LU - x_lib||_2:", norm_x_diff)
    print("||x_LU - A_inv_lib * b_init||_2:", norm_xLU_Ainvlib_diff)


if __name__ == "__main__":
    main()