import pprint
import numpy as np
import scipy.linalg as la
import lib

def generate_vectors(n: int):
    """Generates a Symmetric Positive Definite matrix A and vector b."""
    B = np.random.rand(n, n)
    a = np.dot(B, B.T)
    b = np.random.rand(n)
    return a, b

def compute_determinant(d: np.ndarray):
    """Computes det(A) efficiently using the diagonal matrix D."""
    return np.prod(d)

def _multiply_matrix_vector(a_mod: np.ndarray, x: np.ndarray):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        for j in range(n):
            # If j >= i, original value is in the upper triangle. else we use symmetry.
            a_val = a_mod[i, j] if j >= i else a_mod[j, i]
            y[i] += a_val * x[j]
    return y

def print_decomp(pivots, l, u):
    print("=" * 74)
    pprint.pprint(pivots)
    pprint.pprint(l)
    pprint.pprint(u)
    print("=" * 74)

def main():
    n = int(input("Enter size n: "))
    eps = 10**(-int(input("Enter m for epsilon (10^-m): ")))
    a, b = generate_vectors(n)
    
    pivots, l, u, x_lib = lib.lu_decomp(a, b)
    print_decomp(pivots, l, u)
    
    a_mod, d, x_chol = lib.ldl_decomp(a, b, eps)
    print(f"Determinant: {compute_determinant(d):.4e}")
    
    ax_product = _multiply_matrix_vector(a_mod, x_chol)
    norm_res = la.norm(ax_product - b)
    print(f"Norm 1 (Residual): {norm_res:.2e}")
    
    norm_diff = la.norm(x_chol - x_lib)
    print(f"Norm 2 (Lib Diff): {norm_diff:.2e}")
    
    # Norms should be smaller than 10^-8.
    status = "SUCCESS" if norm_res < 1e-8 and norm_diff < 1e-8 else "FAILURE"
    print(f"Verification Status: {status}")

if __name__ == "__main__":
    main()
