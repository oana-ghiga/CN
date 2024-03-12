import numpy as np

def lu_decomposition(A, eps):
    n = len(A)
    L = np.zeros((n,n))
    U = np.zeros((n,n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
            if i == j:
                L[i, i] = 1
            else:
                if np.abs(U[i, i]) > eps:
                    L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
                else:
                    print("Division cannot be done")
                    return None, None
    return L, U

def forward_substitution(L, b, eps):
    n = len(b)
    y = np.zeros_like(b, dtype=float)

    for i in range(n):
        if np.abs(L[i, i]) > eps:
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        else:
            print("Division cannot be done")
            return None

    return y

def backward_substitution(U, y, eps):
    n = len(y)
    x = np.zeros_like(y, dtype=float)

    for i in range(n - 1, -1, -1):
        if np.abs(U[i, i]) > eps:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        else:
            print("Division cannot be done")
            return None

    return x

def check_solution(A_init, b_init, x_LU):
    norm = np.linalg.norm(A_init @ x_LU - b_init, ord=2)
    return norm

def main():
    # Input data
    n = int(input("Enter the size of the system (n): "))
    t = int(input("Enter the precision of the calculations (t): "))
    epsilon = 10 ** (-t)

    # Generate random matrix A and vector b for testing
    A_init = np.array([[1,2,1,1], [1,4,-1,7], [4,9,5,11],[1,0,6,4]])
    b_init = np.array([0,20,18,1])

    # LU decomposition
    L, U = lu_decomposition(A_init, epsilon)
    if L is None or U is None:
        return

    # Solve the system using LU decomposition
    y = forward_substitution(L, b_init, epsilon)
    if y is None:
        return
    x_LU = backward_substitution(U, y, epsilon)
    if x_LU is None:
        return

    # Calculate the determinant of A
    determinat = np.prod(np.diag(U))*np.prod(np.diag(L))

    # Check the solution
    norm_check = check_solution(A_init, b_init, x_LU)
    print(f"Norm check: {norm_check}")

    # Calculate the solution using numpy's built-in function and the inverse of A
    x_lib = np.linalg.solve(A_init, b_init)
    A_inv_lib = np.linalg.inv(A_init)

    # Calculate the norms
    norm_x_diff = np.linalg.norm(x_LU - x_lib, ord=2)
    norm_xLU_Ainvlib_diff = np.linalg.norm(x_LU - A_inv_lib @ b_init, ord=2)

    # Display results
    print("\nResults:")
    print("LU Decomposition:")
    print("L matrix:")
    print(L)
    print("\nU matrix:")
    print(U)

    print("\n Determinant of L:", np.prod(np.diag(L)))
    print("\n Determinant of U:", np.prod(np.diag(U)))
    print("#######################################################################")
    print("\n Determinant of A:")
    print(determinat)

    print("\nSolution using LU decomposition (x_LU):", x_LU)
    print("\nSolution using NumPy library (x_lib):", x_lib)
    print("\nInverse of A using NumPy library (A_inv_lib):", A_inv_lib)


    print("\nNorms:")
    print("||x_LU - x_lib||_2:", norm_x_diff)
    print("||x_LU - ((A^-1)_lib)*b_init||_2:", norm_xLU_Ainvlib_diff)

if __name__ == "__main__":
    main()
