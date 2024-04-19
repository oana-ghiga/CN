import numpy as np
from scipy.linalg import svd, cholesky
from numpy.linalg import pinv, norm


def jacobi_method(v, n, eps):
    max_iter = 1000
    V = np.eye(n)
    
    def idx(i, j):
        # function to map a matrix index to the index in the vector v
        if i < j:
            i, j = j, i
        return i*(i+1)//2 + j

    for _ in range(max_iter):
        # find largest off-diagonal element A[p][q]
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(v[idx(i, j)]) > abs(v[idx(p, q)]):
                    p, q = i, j

        # convergence
        if abs(v[idx(p, q)]) < eps:
            break

        # calculate rotation to get zeros on off-diagonal
        d = (v[idx(q, q)] - v[idx(p, p)]) / (2.0 * v[idx(p, q)])
        t = np.sign(d) / (abs(d) + np.sqrt(1.0 + d*d))
        c = 1.0 / np.sqrt(1.0 + t*t)
        s = c * t

        # update eigenvalues
        tau = s / (1.0 + c)
        temp = v[idx(p, q)]
        v[idx(p, q)] = 0.0
        v[idx(p, p)] -= t * temp
        v[idx(q, q)] += t * temp
        for r in range(p):      # i < p
            temp = v[idx(r, p)]
            v[idx(r, p)] -= s * (v[idx(r, q)] + tau * temp)
            v[idx(r, q)] += s * (temp - tau * v[idx(r, q)])
        for r in range(p+1, q):  # p < i < q
            temp = v[idx(p, r)]
            v[idx(p, r)] -= s * (v[idx(r, q)] + tau * v[idx(p, r)])
            v[idx(r, q)] += s * (temp - tau * v[idx(r, q)])
        for r in range(q+1, n):  # i > q
            temp = v[idx(p, r)]
            v[idx(p, r)] -= s * (v[idx(q, r)] + tau * temp)
            v[idx(q, r)] += s * (temp - tau * v[idx(q, r)])

        for r in range(n):
            temp = V[r, p]
            V[r, p] -= s * (V[r, q] + tau * V[r, p])
            V[r, q] += s * (temp - tau * V[r, q])

    # eigenvalues are on the diagonal of A
    eigenvalues = np.array([v[idx(i, i)] for i in range(n)])
    # eigenvectors are the columns of V
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
    rank = np.sum(s > 1e-10)
    cond_num = s[0] / s[-1]
    A_inv = Vt.T @ np.diag(1/s) @ U.T
    A_pinv = pinv(A)
    norm_diff = norm(A_inv - A_pinv)
    return s, rank, cond_num, A_inv, A_pinv, norm_diff

def main():
    n = int(input("Enter the size of the matrix (n): "))
    eps = float(input("Enter the computation error (eps): "))
    A = np.random.rand(n, n)
    A = A @ A.T  # make A symmetric

    n = len(A)
    v = A.flatten() # stores the elements from the lower triangular part of the matrix A, bcs A is symmetric
    Lambda, U = jacobi_method(v, n, eps)
    print("Eigenvalues:", Lambda)
    print("Eigenvectors:", U)
    print("Verification norm:", verify_eigen(A, U, np.diag(Lambda)))

    A_last = cholesky_sequence(A, eps, 100)
    print("Last matrix in Cholesky sequence:", A_last)

    if A.shape[0] > A.shape[1]:
        s, rank, cond_num, A_inv, A_pinv, norm_diff = svd_analysis(A)
        print("Singular values:", s)
        print("Rank:", rank)
        print("Conditioning number:", cond_num)
        print("Moore-Penrose pseudo-inverse:", A_inv)
        print("Least squares pseudo-inverse:", A_pinv)
        print("Norm of difference:", norm_diff)

if __name__ == "__main__":
    main()