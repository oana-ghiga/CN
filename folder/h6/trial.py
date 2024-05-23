import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Function to calculate finite differences using Aitken's scheme
def calculate_aitken_scheme(x_vals, y_vals):
    num_points = len(x_vals)
    finite_diff_table = np.zeros((num_points, num_points))
    finite_diff_table[:, 0] = y_vals

    for j in range(1, num_points):
        for i in range(num_points - j):
            finite_diff_table[i, j] = (finite_diff_table[i + 1, j - 1] - finite_diff_table[i, j - 1]) / (x_vals[i + j] - x_vals[i])

    return finite_diff_table

# Function to calculate progressive Newton formula
def newton_interpolation(x_vals, y_vals, target_x, finite_diff_table):
    num_points = len(x_vals)
    interpolated_value = y_vals[0]
    for j in range(1, num_points):
        term = finite_diff_table[0, j]
        for i in range(j):
            term *= (target_x - x_vals[i])
        interpolated_value += term
    return interpolated_value

# Function to perform polynomial interpolation using least squares method
def least_squares_interpolation(x_vals, y_vals, target_x, degree):
    vandermonde_matrix = np.vander(x_vals, degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(vandermonde_matrix, y_vals, rcond=None)[0]
    return np.polyval(coefficients[::-1], target_x)

# Generate equidistant nodes and their corresponding function values
def generate_nodes_and_values(start_x, end_x, num_nodes, function):
    x_vals = np.linspace(start_x, end_x, num_nodes + 1)
    y_vals = function(x_vals)
    return x_vals, y_vals

# Example function f(x)
def example_function(x):
    return np.sin(x)

# Function to handle button click event
def on_interpolate_click():
    try:
        # Read inputs from GUI entry fields
        start_x = float(entry_x0.get())
        end_x = float(entry_xn.get())
        num_nodes = int(entry_n.get())
        target_x = float(entry_target_x.get())

        # Generate equidistant nodes and their corresponding function values
        x_vals, y_vals = generate_nodes_and_values(start_x, end_x, num_nodes, example_function)

        # Perform interpolation using progressive Newton formula and Aitken's scheme
        finite_diff_table = calculate_aitken_scheme(x_vals, y_vals)
        interpolated_value_newton = newton_interpolation(x_vals, y_vals, target_x, finite_diff_table)
        messagebox.showinfo("Interpolation Result", f"Interpolated value using progressive Newton formula: {interpolated_value_newton}")

        # Perform polynomial interpolation using least squares method
        degree_values = range(1, 6)
        for degree in degree_values:
            interpolated_value_polynomial = least_squares_interpolation(x_vals, y_vals, target_x, degree)
            messagebox.showinfo(f"Interpolation Result (Degree {degree})", f"Interpolated value using polynomial of degree {degree}: {interpolated_value_polynomial}")

        # Plotting (optional)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'bo', label='Original Data')
        plt.plot(target_x, example_function(target_x), 'ro', label='True Value')
        plt.axhline(interpolated_value_newton, color='g', linestyle='--', label='Progressive Newton')
        plt.plot(degree_values, [least_squares_interpolation(x_vals, y_vals, target_x, degree) for degree in degree_values], 'go-', label='Polynomial Interpolation')
        plt.title('Interpolation Comparison')
        plt.xlabel('Degree of Polynomial')
        plt.ylabel('Interpolated Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create GUI window
root = tk.Tk()
root.title("Interpolation App")

# Create input fields
tk.Label(root, text="x0:").grid(row=0, column=0)
entry_x0 = tk.Entry(root)
entry_x0.grid(row=0, column=1)

tk.Label(root, text="xn:").grid(row=1, column=0)
entry_xn = tk.Entry(root)
entry_xn.grid(row=1, column=1)

tk.Label(root, text="n (number of nodes):").grid(row=2, column=0)
entry_n = tk.Entry(root)
entry_n.grid(row=2, column=1)

tk.Label(root, text="target x:").grid(row=3, column=0)
entry_target_x = tk.Entry(root)
entry_target_x.grid(row=3, column=1)

# Create button to perform interpolation
btn_interpolate = tk.Button(root, text="Interpolate", command=on_interpolate_click)
btn_interpolate.grid(row=4, column=0, columnspan=2, pady=10)

# Start GUI event loop
root.mainloop()
