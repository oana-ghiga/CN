import tkinter as tk
from tkinter import ttk
import numpy as np

def compute_qr_decomposition(matrix):
    rows, cols = matrix.shape
    Q = np.eye(rows)
    R = np.copy(matrix)

    for i in range(cols - (rows == cols)):
        H = np.eye(rows)
        H[i:, i:] = construct_householder(R[i:, i])
        Q = np.dot(Q, H)
        R = np.dot(H, R)

    return Q, R

def construct_householder(vector):
    v = vector / (vector[0] + np.copysign(np.linalg.norm(vector), vector[0]))
    v[0] = 1
    H = np.eye(len(vector))
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def find_approximate_limit(matrix, tolerance=1e-6):
    current_matrix = matrix.copy()
    norm_diff = float('inf')
    iteration_count = 0

    while norm_diff > tolerance:
        Q, R = compute_qr_decomposition(current_matrix)
        next_matrix = np.dot(R, Q)
        norm_diff = np.linalg.norm(next_matrix - current_matrix)
        current_matrix = next_matrix
        iteration_count += 1

    return current_matrix, iteration_count

def compute_result():
    A = np.array([[float(matrix_entries[i][j].get()) for j in range(matrix_size)] for i in range(matrix_size)])
    approx_limit, iterations = find_approximate_limit(A)
    result_string.set(f"Approximate Limit of A^(k+1):\n{approx_limit}\nNumber of Iterations (k): {iterations}")

    # Hide input widgets
    for row in matrix_entries:
        for entry in row:
            entry.grid_remove()
    size_combobox.grid_remove()
    compute_button.grid_remove()
    size_label.grid_remove()

def initialize_entry_widgets():
    global matrix_entries
    matrix_entries = []
    for i in range(matrix_size):
        row = []
        for j in range(matrix_size):
            entry = ttk.Entry(main_frame, width=8)
            entry.grid(row=i+1, column=j, padx=3, pady=3)
            row.append(entry)
        matrix_entries.append(row)

def clear_result_display():
    result_string.set("")

def on_size_change(event):
    global matrix_size
    matrix_size = int(size_combobox.get())
    initialize_entry_widgets()
    clear_result_display()

# Create main window
root = tk.Tk()
root.title("Matrix Limit Calculator")

# Frame for input
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matrix size selection
size_label = ttk.Label(main_frame, text="Matrix Size:")
size_label.grid(row=0, column=0, sticky=tk.W)
size_options = [str(i) for i in range(2, 6)]
size_combobox = ttk.Combobox(main_frame, values=size_options, state="readonly")
size_combobox.current(0)
size_combobox.grid(row=0, column=1, sticky=tk.W)
size_combobox.bind("<<ComboboxSelected>>", on_size_change)

# Button to calculate
compute_button = ttk.Button(main_frame, text="Calculate", command=compute_result)
compute_button.grid(row=0, column=2, padx=5)

# Result display
result_string = tk.StringVar()
result_label = ttk.Label(main_frame, textvariable=result_string, wraplength=400)
result_label.grid(row=1, column=0, columnspan=3, pady=10)

# Set default matrix size
matrix_size = 0
initialize_entry_widgets()

root.mainloop()
