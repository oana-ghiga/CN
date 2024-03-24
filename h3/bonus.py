import tkinter as tk
from tkinter import ttk
import numpy as np

def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)

    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(R[i:, i])
        Q = np.dot(Q, H)
        R = np.dot(H, R)

    return Q, R

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def approximate_limit(A, epsilon=1e-6):
    A_k = A.copy()
    diff_norm = float('inf')
    k = 0

    while diff_norm > epsilon:
        Q, R = qr_decomposition(A_k)
        A_k1 = np.dot(R, Q)
        diff_norm = np.linalg.norm(A_k1 - A_k)
        A_k = A_k1
        k += 1

    return A_k, k

def calculate():
    A = np.array([[float(A_entry[i][j].get()) for j in range(n)] for i in range(n)])
    approx_limit, k = approximate_limit(A)
    result_text.set(f"Approximate Limit of A^(k+1):\n{approx_limit}\nNumber of Iterations to Reach Approximate Limit (k): {k}")

    # Hide input widgets
    for row in A_entry:
        for entry in row:
            entry.grid_remove()
    matrix_size_combobox.grid_remove()
    calculate_button.grid_remove()
    matrix_label.grid_remove()

def create_entry_widgets():
    global A_entry
    A_entry = []
    for i in range(n):
        row = []
        for j in range(n):
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=i+1, column=j, padx=3, pady=3)
            row.append(entry)
        A_entry.append(row)

def clear_result():
    result_text.set("")

def on_matrix_size_change(event):
    global n
    n = int(matrix_size_combobox.get())
    create_entry_widgets()
    clear_result()


# Create main window
root = tk.Tk()
root.title("Matrix Approximate Limit Calculator")

# Frame for input
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Matrix size selection
matrix_label = ttk.Label(frame, text="Matrix Size:")
matrix_label.grid(row=0, column=0, sticky=tk.W)
matrix_sizes = [str(i) for i in range(2, 6)]
matrix_size_combobox = ttk.Combobox(frame, values=matrix_sizes, state="readonly")
matrix_size_combobox.current(0)
matrix_size_combobox.grid(row=0, column=1, sticky=tk.W)
matrix_size_combobox.bind("<<ComboboxSelected>>", on_matrix_size_change)

# Button to calculate
calculate_button = ttk.Button(frame, text="Calculate", command=calculate)
calculate_button.grid(row=0, column=2, padx=5)

# Result display
result_text = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_text, wraplength=400)
result_label.grid(row=1, column=0, columnspan=3, pady=10)

# Set default matrix size
n = 0
create_entry_widgets()


root.mainloop()
