import numpy as np

def generate_symmetric_matrix(n):
    A = np.random.rand(n, n)
    return 0.5 * (A + A.T)  # Ensures A is symmetric

def householder_qr_decomposition(A):
    n = A.shape[0]
    R = np.copy(A)
    Q = np.eye(n)

    for k in range(n - 1):
        x = R[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x)
        v = v + x
        v = v / np.linalg.norm(v)

        R[k:, k:] = R[k:, k:] - 2 * np.outer(v, np.dot(v, R[k:, k:]))
        Q[k:, :] = Q[k:, :] - 2 * np.outer(v, np.dot(v, Q[k:, :]))

    return Q, R

def calculate_approximate_limit(A, epsilon):
    A_prev = np.copy(A)
    A_next = np.copy(A)

    while True:
        Q, R = householder_qr_decomposition(A_next)
        A_next = np.dot(R, Q)

        if np.linalg.norm(A_next - A_prev) <= epsilon:
            break

        A_prev = np.copy(A_next)

    return A_next

def main():
    n = int(input("Enter the size n of the symmetric matrix: "))
    epsilon = float(input("Enter the epsilon value: "))

    A = generate_symmetric_matrix(n)

    # Task 1
    print("Original Symmetric Matrix A:")
    print(A)

    # Task 2-4
    approximate_limit_matrix = calculate_approximate_limit(A, epsilon)
    print("\nApproximate Limit Matrix A(k+1):")
    print(approximate_limit_matrix)

if __name__ == "__main__":
    main()
