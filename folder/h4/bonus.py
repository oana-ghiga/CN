import numpy as np
from scipy.sparse import csc_matrix, find

def parse_line(line):
    values = line.strip().split(',')
    if len(values) == 3:
        return map(float, values)
    return None

def read_csc_matrix(filename):
    data, rows, cols = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                val, i, j = parsed
                data.append(val)
                rows.append(int(i))
                cols.append(int(j))
    max_row = max(rows) + 1 if rows else 0
    max_col = max(cols) + 1 if cols else 0
    return csc_matrix((data, (rows, cols)), shape=(max_row, max_col))

def add_csc_matrices(A, B):
    return A + B

def verify_csc_matrices_equality(A, B, epsilon=1e-6):
    diff = (A != B)
    if diff.nnz == 0:
        return True
    rows, cols, data = find(diff)
    for r, c in zip(rows, cols):
        if abs(A[r, c] - B[r, c]) >= epsilon:
            return False
    return True

def main():
    filenames = ['a.txt', 'b.txt', 'aplusb.txt']
    matrices = [read_csc_matrix(f) for f in filenames]

    A, B, AplusB = matrices
    C = add_csc_matrices(A, B)

    if verify_csc_matrices_equality(C, AplusB):
        print("The computed sum of matrices from files a.txt and b.txt is equal to the matrix from file aplusb.txt.")
    else:
        print("The computed sum of matrices from files a.txt and b.txt is not equal to the matrix from file aplusb.txt.")

if __name__ == "__main__":
    main()
