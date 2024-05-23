import numpy as np

def obtain_epsilon():
    while True:
        try:
            precision = int(input("Enter precision (t, for epsilon=10^(-t)): "))
            if precision < 5:
                print("Precision must be t >= 5 (epsilon <= 1e-5).")
            else:
                return 10**(-precision)
        except ValueError:
            print("Invalid input. Please enter an integer.")

def is_near_zero(value, epsilon):
    return abs(value) <= epsilon

def generate_random_matrix_and_vector(size):
    matrix = np.random.rand(size, size)
    vector = np.random.rand(size)
    return matrix, vector

def compute_vector_b(matrix, vector):
    size = len(vector)
    result_vector = np.zeros(size)
    for i in range(size):
        result_vector[i] = np.sum(vector[j] * matrix[i, j] for j in range(size))
    return result_vector

def perform_householder_qr(matrix):
    size = matrix.shape[0]
    R = np.copy(matrix)
    Q = np.eye(size)

    for k in range(size - 1):
        x = R[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x)
        v += x
        v /= np.linalg.norm(v)

        R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))
        Q[k:, :] -= 2 * np.outer(v, np.dot(v, Q[k:, :]))

    return Q, R

def solve_system_via_qr(QR, b):
    Q, R = QR
    y = np.dot(Q.T, b)
    x = np.linalg.solve(R, y)
    return x

def calculate_error_metrics(A_initial, b_initial, x_householder, x_qr, s):
    error_diff_qr_householder = np.linalg.norm(x_qr - x_householder, ord=2)
    error_Ax_householder = np.linalg.norm(np.dot(A_initial, x_householder) - b_initial, ord=2)
    error_Ax_qr = np.linalg.norm(np.dot(A_initial, x_qr) - b_initial, ord=2)
    relative_error_householder = np.linalg.norm(x_householder - s, ord=2) / np.linalg.norm(s, ord=2)
    relative_error_qr = np.linalg.norm(x_qr - s, ord=2) / np.linalg.norm(s, ord=2)

    return error_diff_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr

def compute_inverse(matrix):
    Q, R = perform_householder_qr(matrix)
    A_inv_householder = np.linalg.solve(R, Q.T)
    A_inv_builtin = np.linalg.inv(matrix)
    return A_inv_householder, A_inv_builtin

def compare_inverses(A_inv_householder, A_inv_builtin):
    norm_diff = np.linalg.norm(A_inv_householder - A_inv_builtin, ord=2)
    return norm_diff

def construct_reflection_matrix(vector):
    size = len(vector)
    identity = np.eye(size)
    vector = vector.reshape(size, 1)
    H = identity - 2 * np.dot(vector, vector.T)
    return H

def householder_qr_using_reflection(matrix):
    size = matrix.shape[0]
    R = np.copy(matrix)
    Q = np.eye(size)

    for k in range(size - 1):
        x = R[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x)
        v += x
        v /= np.linalg.norm(v)

        H = construct_reflection_matrix(v)
        padded_H = np.eye(size)
        padded_H[k:, k:] = H

        R = np.dot(padded_H, R)
        Q = np.dot(Q, padded_H.T)

    return Q, R

def main():
    size = int(input("Enter the size n of the data: "))
    epsilon = obtain_epsilon()
    print("Using precision (epsilon):", epsilon)
    A_initial, s = generate_random_matrix_and_vector(size)
    b_initial = compute_vector_b(A_initial, s)
    print("Matrix A_initial:", A_initial)
    print("Vector s:", s)
    print("Vector b_initial:", b_initial)

    # Task 1: Householder QR decomposition with reflection matrices
    print("Householder QR decomposition with reflection matrices:")
    QR_reflection = householder_qr_using_reflection(A_initial)
    print("Q:\n", QR_reflection[0])
    print("R:\n", QR_reflection[1])

    # Task 2: Householder QR decomposition
    QR_householder = perform_householder_qr(A_initial)

    # Task 3: Solve the system
    x_qr = solve_system_via_qr(QR_householder, b_initial)
    x_householder = np.linalg.solve(QR_householder[1], np.dot(QR_householder[0].T, b_initial))
    print("Solution of the system:", x_qr)
    error_diff_qr_householder, error_Ax_householder, error_Ax_qr, relative_error_householder, relative_error_qr = calculate_error_metrics(A_initial, b_initial, x_householder, x_qr, s)

    # Task 4: Print errors
    print("Error ||x_QR - x_Householder||2:", error_diff_qr_householder)
    print("Error ||A_initial * x_Householder - b_initial||2:", error_Ax_householder)
    print("Error ||A_initial * x_QR - b_initial||2:", error_Ax_qr)
    print("Relative Error ||x_Householder - s||2 / ||s||2:", relative_error_householder)
    print("Relative Error ||x_QR - s||2 / ||s||2:", relative_error_qr)

    # Task 5: Compute and compare inverses
    A_inv_householder, A_inv_builtin = compute_inverse(A_initial)
    norm_diff = compare_inverses(A_inv_householder, A_inv_builtin)
    print("Norm ||A_inv_householder - A_inv_builtin||2:", norm_diff)

if __name__ == "__main__":
    main()
