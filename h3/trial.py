import numpy as np

def get_epsilon():
    while True:
        try:
            t = int(input("Enter precision (t, for epsilon=10^(-t)): "))
            if t < 5:
                print("Precision must be t >= 5 (epsilon <= 1e-5).")
            else:
                return 10**(-t)
        except ValueError:
            print("Invalid input. Please enter an integer.")

def generate_random_data(n):
    A = np.random.rand(n, n)
    s = np.random.rand(n)
    return A, s

def calculate_vector_b(A, s):
    return np.dot(A, s)

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

        H = np.eye(n)
        H[k:, k:] -= 2 * np.outer(v, v)

        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

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

    # Task 2
    QR_Householder = householder_qr_decomposition(A_init)

    # Task 3
    x_QR = solve_linear_system(QR_Householder, b_init)
    x_Householder = np.linalg.solve(QR_Householder[1], np.dot(QR_Householder[0].T, b_init))

    tolerance = 1e-14

    # Task 4
    error_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr = calculate_errors(
        A_init, b_init, x_Householder, x_QR, s)
    error_qr_householder_msg = f"Error ||x_QR - x_Householder||2: {error_qr_householder:.16e}" if error_qr_householder > tolerance else "Error ||x_QR - x_Householder||2: 0"
    error_Ax_householder_msg = f"Error ||(A_init)*(x_Householder) - (b_init)||2: {error_Ax_householder:.16e}" if error_Ax_householder > tolerance else "Error ||(A_init)*(x_Householder) - (b_init)||2: 0"
    error_Ax_qr_msg = f"Error ||(A_init)*(x_QR) - (b_init)||2: {error_Ax_qr:.16e}" if error_Ax_qr > tolerance else "Error ||(A_init)*(x_QR) - (b_init)||2: 0"
    relative_error_householder_msg = f"Relative Error ||x_Householder - s||2 / ||s||2: {relative_error_householder:.16e}" if relative_error_householder > tolerance else "Relative Error ||x_Householder - s||2 / ||s||2: 0"
    relative_error_qr_msg = f"Relative Error ||x_QR - s||2 / ||s||2: {relative_error_qr:.16e}" if relative_error_qr > tolerance else "Relative Error ||x_QR - s||2 / ||s||2: 0"

    A_inv_householder, A_inv_bibl = calculate_inverse(A_init)
    norm_diff = np.linalg.norm(A_inv_householder - A_inv_bibl, ord=2)
    norm_diff_msg = f"Norm ||(A_Householder)^-1 - (A_bibl)^-1||2: {norm_diff:.16e}" if norm_diff > tolerance else "Norm ||(A_Householder)^-1 - (A_bibl)^-1||2: 0"

    print(error_qr_householder_msg)
    print(error_Ax_householder_msg)
    print(error_Ax_qr_msg)
    print(relative_error_householder_msg)
    print(relative_error_qr_msg)
    print(norm_diff_msg)

if __name__ == "__main__":
    main()
