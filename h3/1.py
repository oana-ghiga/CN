import numpy as np

def get_epsilon():
  """Prompts user for precision (epsilon) and validates input."""
  while True:
    try:
      t = int(input("Enter precision (t, for epsilon=10^(-t)): "))
      if t < 5:
        print("Precision must be t >= 5 (epsilon <= 1e-5).")
      else:
        return 10**(-t)
    except ValueError:
      print("Invalid input. Please enter an integer.")

def is_zero(x, eps):
  """Checks if a number is zero within the given precision."""
  return abs(x) <= eps

def generate_random_data(n):
    A = np.random.rand(n, n)
    s = np.random.rand(n)
    return A, s

def calculate_vector_b(A, s):
    n = len(s)
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.sum(np.fromiter((s[j] * A[i, j] for j in range(n)), dtype=float))
    return b


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

def solve_linear_system(QR, b):
    Q, R = QR
    y = np.dot(Q.T, b)
    x = np.linalg.solve(R, y)
    return x

def calculate_errors(A_init, b_init, x_Householder, x_QR, s):
    error_qr_householder = np.linalg.norm(x_QR - x_Householder, ord=2)
    error_Ax_householder = np.linalg.norm(np.dot(A_init, x_Householder) - b_init, ord=2)
    error_Ax_qr = np.linalg.norm(np.dot(A_init, x_QR) - b_init, ord=2)
    relative_error_householder = np.linalg.norm(x_Householder - s, ord=2) / np.linalg.norm(s, ord=2)
    relative_error_qr = np.linalg.norm(x_QR - s, ord=2) / np.linalg.norm(s, ord=2)

    return error_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr

def calculate_inverse(A):
    Q, R = householder_qr_decomposition(A)
    A_inv_householder = np.linalg.solve(R, Q.T)

    A_inv_bibl = np.linalg.inv(A)

    return A_inv_householder, A_inv_bibl

def compare_inverse_norm(A_inv_householder, A_inv_bibl):
    norm_diff = np.linalg.norm(A_inv_householder - A_inv_bibl, ord=2)
    return norm_diff

def main():
    n = int(input("Enter the size n of the data: "))
    eps = get_epsilon()
    print("Using precision (epsilon):", eps)
    A_init, s = generate_random_data(n)
    b_init = calculate_vector_b(A_init, s)
    print("Matrix A_init:", A_init)
    print("Vector s:", s)
    print("Vector b_init:", b_init)


    # Task 2
    QR_Householder = householder_qr_decomposition(A_init)

    # Task 3
    x_QR = solve_linear_system(QR_Householder, b_init)
    x_Householder = np.linalg.solve(QR_Householder[1], np.dot(QR_Householder[0].T, b_init))
    print("Solution of the system:",x_QR)
    error_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr = calculate_errors(A_init, b_init, x_Householder, x_QR, s)

    # Task 4
    print(f"Error ||x_QR - x_Householder||2: {error_qr_householder}")
    print(f"Error ||(A_init)*(x_Householder) - (b_init)||2: {error_Ax_householder}")
    print(f"Error ||(A_init)*(x_QR) - (b_init)||2: {error_Ax_qr}")
    print(f"Relative Error ||x_Householder - s||2 / ||s||2: {relative_error_householder}")
    print(f"Relative Error ||x_QR - s||2 / ||s||2: {relative_error_qr}")

    # Task 5
    A_inv_householder, A_inv_bibl = calculate_inverse(A_init)
    norm_diff = compare_inverse_norm(A_inv_householder, A_inv_bibl)
    print(f"Norm ||(A_Householder)^-1 - (A_bibl)^-1||2: {norm_diff}")

if __name__ == "__main__":
    main()
