import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

def perform_lu_decomposition(matrix):
    size = len(matrix)
    lower = np.eye(size)
    upper = matrix.copy()

    for pivot in range(size - 1):
        for row in range(pivot + 1, size):
            multiplier = upper[row, pivot] / upper[pivot, pivot]
            lower[row, pivot] = multiplier
            upper[row, pivot:] -= multiplier * upper[pivot, pivot:]

    return lower, upper

def calculate_determinant(lower, upper):
    return np.prod(np.diag(lower)) * np.prod(np.diag(upper))

def solve_forward_substitution(lower, vector):
    size = len(vector)
    solution = np.zeros_like(vector, dtype=float)

    for i in range(size):
        solution[i] = (vector[i] - np.dot(lower[i, :i], solution[:i])) / lower[i, i]

    return solution

def solve_backward_substitution(upper, vector):
    size = len(vector)
    solution = np.zeros_like(vector, dtype=float)

    for i in range(size - 1, -1, -1):
        solution[i] = (vector[i] - np.dot(upper[i, i + 1:], solution[i + 1:])) / upper[i, i]

    return solution

def validate_solution(original_matrix, original_vector, computed_solution):
    error_norm = np.linalg.norm(original_matrix @ computed_solution - original_vector, ord=2)
    return error_norm

class LUDecompositionApp:
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
            size = int(self.entry_size.get())
            epsilon = float(self.entry_precision.get())

            # Generate random matrix and vector for testing
            matrix = np.random.rand(size, size)
            vector = np.random.rand(size)

            # Perform LU decomposition
            lower, upper = perform_lu_decomposition(matrix)

            # Solve the system using LU decomposition
            intermediate_solution = solve_forward_substitution(lower, vector)
            final_solution = solve_backward_substitution(upper, intermediate_solution)

            # Validate the solution
            error_norm = validate_solution(matrix, vector, final_solution)

            # Display results in a new window
            result_window = tk.Toplevel(self.root)
            result_window.title("LU Decomposition Results")

            result_label = ttk.Label(result_window, text=f"Solution error norm: {error_norm}")
            result_label.grid(column=0, row=0, pady=10)

            # Add more labels or widgets to display additional results as needed

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

def main():
    root = tk.Tk()
    app = LUDecompositionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
