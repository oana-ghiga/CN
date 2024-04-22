import numpy as np
import matplotlib.pyplot as plt

# Function to calculate finite differences using Aitken's scheme
def aitken(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])

    return F

# Function to calculate progressive Newton formula
def progressive_newton(x, y, target_x, F):
    n = len(x)
    result = y[0]
    for j in range(1, n):
        term = F[0, j]
        for i in range(j):
            term *= (target_x - x[i])
        result += term
    return result

# Function to perform polynomial interpolation using least squares method
def polynomial_interpolation(x, y, target_x, m):
    A = np.vander(x, m + 1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.polyval(coeffs[::-1], target_x)

# Generate equidistant nodes and their corresponding function values
def generate_nodes_and_values(x0, xn, n, f):
    h = (xn - x0) / n
    x = np.linspace(x0, xn, n + 1)
    y = f(x)
    return x, y

# Example function f(x) - replace with your own function
def example_function(x):
    return np.sin(x)

# Main function to perform interpolation and plot results
def main():
    # Read input from the user
    x0 = float(input("Enter x0: "))
    xn = float(input("Enter xn: "))
    n = int(input("Enter the number of equidistant nodes (n): "))

    # Generate equidistant nodes and their corresponding function values
    x, y = generate_nodes_and_values(x0, xn, n, example_function)

    # Choose target_x for interpolation
    target_x = float(input("Enter the value of target x (¯x): "))

    # Perform interpolation using progressive Newton formula and Aitken's scheme
    F = aitken(x, y)
    interpolated_value_newton = progressive_newton(x, y, target_x, F)
    print("Interpolated value using progressive Newton formula:", interpolated_value_newton)
    print("Absolute error using progressive Newton formula:", abs(interpolated_value_newton - example_function(target_x)))

    # Perform polynomial interpolation using least squares method
    m_values = range(1, 6)
    interpolated_values_polynomial = []
    absolute_errors_polynomial = []
    sum_absolute_errors = []
    for m in m_values:
        interpolated_value_polynomial = polynomial_interpolation(x, y, target_x, m)
        interpolated_values_polynomial.append(interpolated_value_polynomial)
        absolute_error_polynomial = abs(interpolated_value_polynomial - example_function(target_x))
        absolute_errors_polynomial.append(absolute_error_polynomial)
        sum_absolute_errors.append(np.sum(np.abs(np.polyval(np.polyfit(x, y, m)[::-1], x) - y)))

        print(f"Interpolated value using polynomial of degree {m}:", interpolated_value_polynomial)
        print(f"Absolute error using polynomial of degree {m}:", absolute_error_polynomial)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bo', label='Original Data')
    plt.plot(target_x, example_function(target_x), 'ro', label='True Value')

    # Plot progressive Newton formula
    plt.axhline(interpolated_value_newton, color='g', linestyle='--', label='Progressive Newton')

    # Plot polynomial interpolation
    plt.plot(m_values, interpolated_values_polynomial, 'go-', label='Polynomial Interpolation')

    plt.title('Interpolation Comparison')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
