import numpy as np

def perform_lu_decomposition(matrix, tolerance):
    size = len(matrix)
    lower = np.zeros((size, size))
    upper = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            upper[i, j] = matrix[i, j] - sum(lower[i, k] * upper[k, j] for k in range(i))
        for j in range(i + 1, size):
            if np.abs(upper[i, i]) > tolerance:
                lower[j, i] = (matrix[j, i] - sum(lower[j, k] * upper[k, i] for k in range(i))) / upper[i, i]
            else:
                print("Division by nearly zero detected")
                return None, None
        lower[i, i] = 1
    return lower, upper

def solve_forward_substitution(lower, vector, tolerance):
    size = len(vector)
    result = np.zeros_like(vector, dtype=float)

    for i in range(size):
        if np.abs(lower[i, i]) > tolerance:
            result[i] = (vector[i] - np.dot(lower[i, :i], result[:i])) / lower[i, i]
        else:
            print("Division by nearly zero detected")
            return None

    return result

def solve_backward_substitution(upper, vector, tolerance):
    size = len(vector)
    result = np.zeros_like(vector, dtype=float)

    for i in range(size - 1, -1, -1):
        if np.abs(upper[i, i]) > tolerance:
            result[i] = (vector[i] - np.dot(upper[i, i + 1:], result[i + 1:])) / upper[i, i]
        else:
            print("Division by nearly zero detected")
            return None

    return result

def verify_solution(initial_matrix, initial_vector, solution):
    error_norm = np.linalg.norm(initial_matrix @ solution - initial_vector, ord=2)
    return error_norm

def main():
    # Gather input data
    size = int(input("Enter the size of the matrix (n): "))
    precision = int(input("Enter the precision of calculations (t): "))
    epsilon = 10 ** (-precision)

    # Test data
    matrix = np.array([[1,2,1,1], [1,4,-1,7], [4,9,5,11], [1,0,6,4]])
    vector = np.array([0, 20, 18, 1])

    # Perform LU decomposition
    lower, upper = perform_lu_decomposition(matrix, epsilon)
    if lower is None or upper is None:
        return

    # Solve system using LU decomposition
    intermediate_vector = solve_forward_substitution(lower, vector, epsilon)
    if intermediate_vector is None:
        return
    solution_lu = solve_backward_substitution(upper, intermediate_vector, epsilon)
    if solution_lu is None:
        return

    # Compute determinant of the matrix
    determinant = np.prod(np.diag(upper)) * np.prod(np.diag(lower))

    # Verify the solution
    solution_error = verify_solution(matrix, vector, solution_lu)
    print(f"Solution verification error norm: {solution_error}")

    # Use numpy's built-in functions to solve and invert
    solution_lib = np.linalg.solve(matrix, vector)
    inverse_lib = np.linalg.inv(matrix)

    # Compute norms
    solution_diff_norm = np.linalg.norm(solution_lu - solution_lib, ord=2)
    inverse_solution_diff_norm = np.linalg.norm(solution_lu - inverse_lib @ vector, ord=2)

    # Display results
    print("\nLU Decomposition Results:")
    print("L matrix:")
    print(lower)
    print("\nU matrix:")
    print(upper)

    print(f"\nDeterminant of the matrix: {determinant}")

    print("\nSolution using LU decomposition (solution_lu):", solution_lu)
    print("\nSolution using NumPy (solution_lib):", solution_lib)
    print("\nInverse of the matrix using NumPy (inverse_lib):", inverse_lib)

    print("\nNorms:")
    print("||solution_lu - solution_lib||_2:", solution_diff_norm)
    print("||solution_lu - (inverse_lib @ vector)||_2:", inverse_solution_diff_norm)

if __name__ == "__main__":
    main()
