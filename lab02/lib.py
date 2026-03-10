import numpy as np
import scipy.linalg as la

def lu_decomp(a: np.ndarray, b: np.ndarray):
    p, l, u = la.lu(a)
    x = la.solve(a, b)
    return p, l, u, x

def _solve(a_mod: np.ndarray, d: np.ndarray, b: np.ndarray):
    n = len(b)
    
    # 1. Forward Substitution.
    z = np.zeros(n)
    for i in range(n):
        z[i] = b[i] - sum(a_mod[i, j] * z[j] for j in range(i))
        
    # 2. Diagonal System: Dy = z.
    y = z / d
    
    # 3. Backward Substitution.
    x_chol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # L^T elements are a_mod[j, i] because L was stored in lower a_mod [cite: 153]
        x_chol[i] = y[i] - sum(a_mod[j, i] * x_chol[j] for j in range(i + 1, n))
        
    return x_chol

def ldl_decomp(a: np.ndarray, b: np.ndarray, eps: float):
    a_mod = a.copy()
    n = a_mod.shape[0]
    d = np.zeros(n)
    
    for p in range(n):
        # Calculate diagonal element d_p.
        sum_d_l2 = sum(d[k] * (a_mod[p, k]**2) for k in range(p))
        d[p] = a_mod[p, p] - sum_d_l2
        
        if abs(d[p]) <= eps:
            raise ValueError(f"Matrix is not positive definite; d[{p}] is zero.")

        # Calculate elements of column p for matrix L.
        for i in range(p + 1, n):
            sum_l_l_d = sum(d[k] * a_mod[i, k] * a_mod[p, k] for k in range(p))
            a_mod[i, p] = (a_mod[i, p] - sum_l_l_d) / d[p]
            
    x_chol = _solve(a_mod, d, b)
    return a_mod, d, x_chol
