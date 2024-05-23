import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

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

# Example function f(x)
def example_function(x):
    return np.sin(x)

# Function to handle button click event
def on_interpolate_click():
    try:
        # Read inputs from GUI entry fields
        x0 = float(entry_x0.get())
        xn = float(entry_xn.get())
        n = int(entry_n.get())
        target_x = float(entry_target_x.get())

        # Generate equidistant nodes and their corresponding function values
        x, y = generate_nodes_and_values(x0, xn, n, example_function)

        # Perform interpolation using progressive Newton formula and Aitken's scheme
        F = aitken(x, y)
        interpolated_value_newton = progressive_newton(x, y, target_x, F)
        messagebox.showinfo("Interpolation Result", f"Interpolated value using progressive Newton formula: {interpolated_value_newton}")

        # Perform polynomial interpolation using least squares method
        m_values = range(1, 6)
        for m in m_values:
            interpolated_value_polynomial = polynomial_interpolation(x, y, target_x, m)
            messagebox.showinfo(f"Interpolation Result (Degree {m})", f"Interpolated value using polynomial of degree {m}: {interpolated_value_polynomial}")

        # Plotting (optional)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'bo', label='Original Data')
        plt.plot(target_x, example_function(target_x), 'ro', label='True Value')
        plt.axhline(interpolated_value_newton, color='g', linestyle='--', label='Progressive Newton')
        plt.plot(m_values, [polynomial_interpolation(x, y, target_x, m) for m in m_values], 'go-', label='Polynomial Interpolation')
        plt.title('Interpolation Comparison')
        plt.xlabel('x')
        plt.ylabel('f(x)')
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
