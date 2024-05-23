import numpy as np
from scipy.linalg import svd, cholesky
from numpy.linalg import pinv, norm

def jacobi_method(v, n, eps):
    def index(i, j):
        if i < j: i, j = j, i
        return i*(i+1)//2 + j

    max_iter = 1000
    V = np.eye(n)
    iters = 0
    while iters < max_iter:
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(v[index(i, j)]) > abs(v[index(p, q)]):
                    p, q = i, j
        if abs(v[index(p, q)]) < eps:
            break

        d = (v[index(q, q)] - v[index(p, p)]) / (2.0 * v[index(p, q)])
        t = np.sign(d) / (abs(d) + np.sqrt(1.0 + d*d))
        c = 1.0 / np.sqrt(1.0 + t*t)
        s = c * t

        tau = s / (1.0 + c)
        temp = v[index(p, q)]
        v[index(p, q)] = 0.0
        v[index(p, p)] -= t * temp
        v[index(q, q)] += t * temp

        for r in range(n):
            if r != p and r != q:
                temp = v[index(min(r, p), max(r, p))]
                v[index(min(r, p), max(r, p))] -= s * (v[index(min(r, q), max(r, q))] + tau * temp)
                v[index(min(r, q), max(r, q))] += s * (temp - tau * v[index(min(r, q), max(r, q))])
        
        for r in range(n):
            temp = V[r, p]
            V[r, p] -= s * (V[r, q] + tau * V[r, p])
            V[r, q] += s * (temp - tau * V[r, q])
        
        iters += 1

    eigenvalues = np.array([v[index(i, i)] for i in range(n)])
    eigenvectors = V.T
    return eigenvalues, eigenvectors

def verify_eigen(A, U, Lambda):
    return norm(A @ U - U @ Lambda)

def cholesky_sequence(A, eps, k_max):
    for k in range(k_max):
        L = cholesky(A)
        A_new = L.T @ L
        if norm(A_new - A) < eps:
            break
        A = A_new
    return A

def svd_analysis(A):
    U, s, Vt = svd(A)
    rank = sum([1 for sv in s if sv > 1e-10])
    cond_num = s[0] / s[-1] if s[-1] != 0 else np.inf
    A_inv = Vt.T @ np.diag(1/s) @ U.T
    A_pinv = pinv(A)
    norm_diff = norm(A_inv - A_pinv)
    return s, rank, cond_num, A_inv, A_pinv, norm_diff

def main():
    n = int(input("enter size of the matrix (n): "))
    eps = float(input("enter computation error (eps): "))
    A = np.random.rand(n, n)
    A = A @ A.T

    flat_A = [A[i, j] if i >= j else A[j, i] for i in range(n) for j in range(i+1)]
    Lambda, U = jacobi_method(flat_A, n, eps)
    print("eigenvalues:", Lambda)
    print("eigenvectors:", U)
    print("verification norm:", verify_eigen(A, U, np.diag(Lambda)))

    A_last = cholesky_sequence(A, eps, 100)
    print("last matrix in Cholesky sequence:", A_last)

    if A.shape[0] > A.shape[1]:
        s, rank, cond_num, A_inv, A_pinv, norm_diff = svd_analysis(A)
        print("singular values:", s)
        print("rank:", rank)
        print("conditioning number:", cond_num)
        print("moore-Penrose pseudo-inverse:", A_inv)
        print("least squares pseudo-inverse:", A_pinv)
        print("norm of difference:", norm_diff)

if __name__ == "__main__":
    main()