import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n*(n+1))//2)
    U = np.zeros((n*(n+1))//2)

    for i in range(n):
        for j in range(i, n):
            U[i*(i+1)//2 + j - i] = A[i][j]
            for k in range(i):
                U[i*(i+1)//2 + j - i] -= L[i*(i+1)//2 + k] * U[k*(k+1)//2 + j - k]

        for j in range(i, n):
            if i == j:
                L[i*(i+1)//2 + i] = 1
            else:
                L[j*(j+1)//2 + i] = A[j][i]
                for k in range(i):
                    L[j*(j+1)//2 + i] -= L[j*(j+1)//2 + k] * U[k*(k+1)//2 + i - k]
                L[j*(j+1)//2 + i] /= U[i*(i+1)//2 + i - i]

    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i*(i+1)//2 + j] * y[j]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i*(i+1)//2 + j - i] * x[j]
        x[i] /= U[i*(i+1)//2 + i - i]
    return x

def solve_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def main():
    n = int(input("Enter the size of the matrix (n): "))
    epsilon = float(input("Enter the value of epsilon: "))

    # Generate a random nxn matrix A and vector b
    A = np.random.rand(n, n)
    b = np.random.rand(n)

    # Add epsilon to the diagonal elements of A to ensure it's invertible
    A += np.eye(n) * epsilon

    x = solve_lu(A, b)

    print("Solution x:")
    print(x)

if __name__ == "__main__":
    main()
