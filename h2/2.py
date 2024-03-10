import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U

def determinant_LU(L, U):
    return np.prod(np.diag(L)) * np.prod(np.diag(U))

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def check_solution(A_init, b_init, x_LU):
    norm = np.linalg.norm(A_init @ x_LU - b_init, ord=2)
    return norm

class LUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LU Decomposition Solver")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.label_size = ttk.Label(self.frame, text="Enter the size of the system (n):")
        self.label_size.grid(column=0, row=0, sticky=tk.W, pady=5)

        self.entry_size = ttk.Entry(self.frame)
        self.entry_size.grid(column=1, row=0, sticky=tk.W, pady=5)

        self.label_precision = ttk.Label(self.frame, text="Enter the precision of the calculations (epsilon):")
        self.label_precision.grid(column=0, row=1, sticky=tk.W, pady=5)

        self.entry_precision = ttk.Entry(self.frame)
        self.entry_precision.grid(column=1, row=1, sticky=tk.W, pady=5)

        self.solve_button = ttk.Button(self.frame, text="Solve", command=self.solve_system)
        self.solve_button.grid(column=0, row=2, columnspan=2, pady=10)

    def solve_system(self):
        try:
            n = int(self.entry_size.get())
            epsilon = float(self.entry_precision.get())

            # Generate random matrix A and vector b for testing
            A_init = np.random.rand(n, n)
            b_init = np.random.rand(n)

            # LU decomposition
            L, U = lu_decomposition(A_init)

            # Solve the system using LU decomposition
            y = forward_substitution(L, b_init)
            x_LU = backward_substitution(U, y)

            # Check the solution
            norm_check = check_solution(A_init, b_init, x_LU)

            # Display results in a new window
            result_window = tk.Toplevel(self.root)
            result_window.title("LU Decomposition Results")

            result_label = ttk.Label(result_window, text=f"Norm check: {norm_check}")
            result_label.grid(column=0, row=0, pady=10)

            # Add more labels or widgets to display additional results as needed

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

def main():
    root = tk.Tk()
    app = LUApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
