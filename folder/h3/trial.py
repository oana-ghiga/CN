import numpy as np


def obtain_epsilon():
    while True:
        try:
            t = int(input("Enter precision (t, for epsilon=10^(-t)): "))
            if t < 5:
                print("Precision must be t >= 5 (epsilon <= 1e-5).")
            else:
                return 10 ** (-t)
        except ValueError:
            print("Invalid input. Please enter an integer.")


def create_random_data(size):
    matrix_A = np.random.rand(size, size)
    vector_s = np.random.rand(size)
    return matrix_A, vector_s


def compute_vector_b(matrix_A, vector_s):
    return np.dot(matrix_A, vector_s)


def perform_householder_qr(matrix):
    size = matrix.shape[0]
    R_matrix = np.copy(matrix)
    Q_matrix = np.eye(size)

    for k in range(size - 1):
        x = R_matrix[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x)
        v = v + x
        v = v / np.linalg.norm(v)

        H = np.eye(size)
        H[k:, k:] -= 2 * np.outer(v, v)

        R_matrix = np.dot(H, R_matrix)
        Q_matrix = np.dot(Q_matrix, H.T)

    return Q_matrix, R_matrix


def solve_system(QR_decomposition, vector_b):
    Q_matrix, R_matrix = QR_decomposition
    intermediate_y = np.dot(Q_matrix.T, vector_b)
    solution_x = np.linalg.solve(R_matrix, intermediate_y)
    return solution_x


def compute_errors(matrix_A_initial, vector_b_initial, x_householder, x_qr, vector_s):
    error_qr_householder = np.linalg.norm(x_qr - x_householder, ord=2)
    error_Ax_householder = np.linalg.norm(np.dot(matrix_A_initial, x_householder) - vector_b_initial, ord=2)
    error_Ax_qr = np.linalg.norm(np.dot(matrix_A_initial, x_qr) - vector_b_initial, ord=2)
    relative_error_householder = np.linalg.norm(x_householder - vector_s, ord=2) / np.linalg.norm(vector_s, ord=2)
    relative_error_qr = np.linalg.norm(x_qr - vector_s, ord=2) / np.linalg.norm(vector_s, ord=2)

    return error_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr


def compute_inverse(matrix):
    Q_matrix, R_matrix = perform_householder_qr(matrix)
    inv_householder = np.linalg.solve(R_matrix, Q_matrix.T)
    inv_builtin = np.linalg.inv(matrix)
    return inv_householder, inv_builtin


def compare_inverses(inv_householder, inv_builtin):
    norm_diff = np.linalg.norm(inv_householder - inv_builtin, ord=2)
    return norm_diff


def main():
    n = int(input("Enter the size n of the data: "))
    epsilon = obtain_epsilon()
    print("Using precision (epsilon):", epsilon)
    A_initial, s_vector = create_random_data(n)
    b_initial = compute_vector_b(A_initial, s_vector)

    # QR decomposition using Householder reflections
    QR_householder = perform_householder_qr(A_initial)

    # Solving the linear system
    x_qr = solve_system(QR_householder, b_initial)
    x_householder = np.linalg.solve(QR_householder[1], np.dot(QR_householder[0].T, b_initial))

    tolerance = 1e-14

    # Calculating errors
    error_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr = compute_errors(
        A_initial, b_initial, x_householder, x_qr, s_vector)

    # Displaying error messages
    print(
        f"Error ||x_QR - x_Householder||2: {error_qr_householder:.16e}" if error_qr_householder > tolerance else "Error ||x_QR - x_Householder||2: 0")
    print(
        f"Error ||(A_init)*(x_Householder) - (b_init)||2: {error_Ax_householder:.16e}" if error_Ax_householder > tolerance else "Error ||(A_init)*(x_Householder) - (b_init)||2: 0")
    print(
        f"Error ||(A_init)*(x_QR) - (b_init)||2: {error_Ax_qr:.16e}" if error_Ax_qr > tolerance else "Error ||(A_init)*(x_QR) - (b_init)||2: 0")
    print(
        f"Relative Error ||x_Householder - s||2 / ||s||2: {relative_error_householder:.16e}" if relative_error_householder > tolerance else "Relative Error ||x_Householder - s||2 / ||s||2: 0")
    print(
        f"Relative Error ||x_QR - s||2 / ||s||2: {relative_error_qr:.16e}" if relative_error_qr > tolerance else "Relative Error ||x_QR - s||2 / ||s||2: 0")

    # Calculating inverses and their norm difference
    A_inv_householder, A_inv_builtin = compute_inverse(A_initial)
    norm_diff = compare_inverses(A_inv_householder, A_inv_builtin)
    print(
        f"Norm ||(A_Householder)^-1 - (A_builtin)^-1||2: {norm_diff:.16e}" if norm_diff > tolerance else "Norm ||(A_Householder)^-1 - (A_builtin)^-1||2: 0")


if __name__ == "__main__":
    main()
