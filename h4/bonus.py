import numpy as np
from scipy.sparse import csc_matrix, find

def read_csc_matrix(filename):
    data = []
    rows = []
    cols = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split(',')
            if len(values) != 3:  # skip lines that do not contain exactly three values
                continue
            val, i, j = map(float, values)
            data.append(val)
            rows.append(int(i))
            cols.append(int(j))
    m = max(rows) + 1 
    n = max(cols) + 1 
    return csc_matrix((data, (rows, cols)), shape=(m, n))
def add_csc_matrices(A, B):
    return A + B

def verify_csc_matrices_equality(A, B, epsilon=1e-6):
    A_rows, A_cols, A_data = find(A)
    B_rows, B_cols, B_data = find(B)
    for a_row, a_col, a_val in zip(A_rows, A_cols, A_data):
        b_val = B[a_row, a_col]
        if abs(a_val - b_val) >= epsilon:
            return False
    return True

if __name__ == "__main__":
    A = read_csc_matrix('a.txt')
    B = read_csc_matrix('b.txt')
    AplusB = read_csc_matrix('aplusb.txt')

    C = add_csc_matrices(A, B)

    if verify_csc_matrices_equality(C, AplusB):
        print("the computed sum of matrices from files a.txt and b.txt is equal to the matrix from file aplusb.txt.")
    else:
        print("the computed sum of matrices from files a.txt and b.txt is not equal to the matrix from file aplusb.txt.")