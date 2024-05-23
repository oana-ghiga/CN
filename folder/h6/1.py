import numpy as np
import matplotlib.pyplot as plt

# Function to calculate finite differences using Aitken's scheme
def aitken_scheme(x_values, y_values):
    num_points = len(x_values)
    finite_diff_table = np.zeros((num_points, num_points))
    finite_diff_table[:, 0] = y_values

    for j in range(1, num_points):
        for i in range(num_points - j):
            finite_diff_table[i, j] = (finite_diff_table[i + 1, j - 1] - finite_diff_table[i, j - 1]) / (x_values[i + j] - x_values[i])

    return finite_diff_table

# Function to calculate progressive Newton formula
def newton_interpolation(x_values, y_values, target_x, finite_diff_table):
    num_points = len(x_values)
    interpolated_value = y_values[0]
    for j in range(1, num_points):
        term = finite_diff_table[0, j]
        for i in range(j):
            term *= (target_x - x_values[i])
        interpolated_value += term
    return interpolated_value

# Function to perform polynomial interpolation using least squares method
def least_squares_interpolation(x_values, y_values, target_x, degree):
    vandermonde_matrix = np.vander(x_values, degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(vandermonde_matrix, y_values, rcond=None)[0]
    return np.polyval(coefficients[::-1], target_x)

# Generate equidistant nodes and their corresponding function values
def generate_nodes_and_values(start_x, end_x, num_nodes, function):
    step_size = (end_x - start_x) / num_nodes
    x_values = np.linspace(start_x, end_x, num_nodes + 1)
    y_values = function(x_values)
    return x_values, y_values

# Example function f(x) - replace with your own function
def example_function(x):
    return np.sin(x)

# Main function
def main():
    # Read input from the user
    start_x = float(input("Enter start x (x0): "))
    end_x = float(input("Enter end x (xn): "))
    num_nodes = int(input("Enter the number of equidistant nodes (n): "))

    # Generate equidistant nodes and their corresponding function values
    x_values, y_values = generate_nodes_and_values(start_x, end_x, num_nodes, example_function)

    # Choose target_x for interpolation
    target_x = float(input("Enter the value of target x (x̄): "))

    # Perform interpolation using progressive Newton formula and Aitken's scheme
    finite_diff_table = aitken_scheme(x_values, y_values)

    interpolated_value_newton = newton_interpolation(x_values, y_values, target_x, finite_diff_table)
    # Read the new x value from the user
    new_x = float(input("Enter the new x value: "))

    # Calculate Ln(new_x) using the progressive Newton formula
    ln_new_x = newton_interpolation(x_values, y_values, new_x, finite_diff_table)

    print("Interpolated value using progressive Newton formula:", interpolated_value_newton)
    print("Absolute error using progressive Newton formula:", abs(interpolated_value_newton - example_function(target_x)))
    print(f"Ln({new_x}) = {ln_new_x}")

    # Perform polynomial interpolation using least squares method
    degree_values = range(1, 6)
    interpolated_values_polynomial = []
    absolute_errors_polynomial = []
    sum_absolute_errors = []
    for degree in degree_values:
        interpolated_value_polynomial = least_squares_interpolation(x_values, y_values, target_x, degree)
        interpolated_values_polynomial.append(interpolated_value_polynomial)
        absolute_error_polynomial = abs(interpolated_value_polynomial - example_function(target_x))
        absolute_errors_polynomial.append(absolute_error_polynomial)
        sum_absolute_errors.append(np.sum(np.abs(np.polyval(np.polyfit(x_values, y_values, degree)[::-1], x_values) - y_values)))

        print(f"Interpolated value using polynomial of degree {degree}:", interpolated_value_polynomial)
        print(f"Absolute error using polynomial of degree {degree}:", absolute_error_polynomial)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'bo', label='Original Data')
    plt.plot(target_x, example_function(target_x), 'ro', label='True Value')

    # Plot progressive Newton formula
    plt.axhline(interpolated_value_newton, color='g', linestyle='--', label='Progressive Newton')

    # Plot polynomial interpolation
    plt.plot(degree_values, interpolated_values_polynomial, 'go-', label='Polynomial Interpolation')

    plt.title('Interpolation Comparison')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Interpolated Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
