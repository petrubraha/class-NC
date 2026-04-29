import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def target_function(x):
    """The unknown function f(x) to be approximated."""
    return x**4 - 12*x**3 + 30*x**2 + 12

def generate_interpolation_nodes(x0, xn, n, func):
    """Generates random sorted interpolation nodes and their function values."""
    inner_nodes = np.sort(np.random.uniform(x0, xn, n - 1))
    x_nodes = np.concatenate(([x0], inner_nodes, [xn]))
    y_nodes = func(x_nodes)
    return x_nodes, y_nodes

def solve_least_squares(x_nodes, y_nodes, m):
    """Computes coefficients for a polynomial of degree m using least squares."""
    # We analyse the Ba = f system.
    matrix_b = np.zeros((m + 1, m + 1))
    vector_f = np.zeros(m + 1)
    
    for i in range(m + 1):
        for j in range(m + 1):
            matrix_b[i, j] = np.sum(x_nodes**(i + j))
        vector_f[i] = np.sum(y_nodes * (x_nodes**i))
    
    coeffs_a = la.solve(matrix_b, vector_f)
    return coeffs_a

def evaluate_polynomial_horner(coeffs, x_val):
    """Evaluates polynomial at x_val using Horner's method."""
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * x_val + coeffs[i]
    return result

def solve_cubic_spline(x_nodes, y_nodes, d_a, d_b):
    """Computes the A coefficients for a C2 cubic spline."""
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    matrix_h = np.zeros((n + 1, n + 1))
    vector_r = np.zeros(n + 1)

    # Boundary condition at x0 (da)
    matrix_h[0, 0] = 2 * h[0]
    matrix_h[0, 1] = h[0]
    vector_r[0] = 6 * ((y_nodes[1] - y_nodes[0]) / h[0] - d_a)

    # Continuity conditions for internal nodes
    for i in range(1, n):
        matrix_h[i, i-1] = h[i-1]
        matrix_h[i, i] = 2 * (h[i-1] + h[i])
        matrix_h[i, i+1] = h[i]
        vector_r[i] = 6 * ((y_nodes[i+1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i-1]) / h[i-1])

    # Boundary condition at xn (db)
    matrix_h[n, n-1] = h[n-1]
    matrix_h[n, n] = 2 * h[n-1]
    vector_r[n] = 6 * (d_b - (y_nodes[n] - y_nodes[n-1]) / h[n-1])

    coeffs_a = la.solve(matrix_h, vector_r)
    return coeffs_a, h

def evaluate_spline(x_val, x_nodes, y_nodes, coeffs_a, h):
    """Evaluates the spline function Sf at x_val."""
    if x_val < x_nodes[0] or x_val > x_nodes[-1]:
        return None
    
    i0 = np.searchsorted(x_nodes, x_val) - 1
    i0 = max(0, min(i0, len(x_nodes) - 2))

    term1 = ((x_val - x_nodes[i0])**3 * coeffs_a[i0+1]) / (6 * h[i0])
    term2 = ((x_nodes[i0+1] - x_val)**3 * coeffs_a[i0]) / (6 * h[i0])
    
    bi0 = (y_nodes[i0+1] - y_nodes[i0]) / h[i0] - (h[i0] * (coeffs_a[i0+1] - coeffs_a[i0])) / 6
    ci0 = (x_nodes[i0+1] * y_nodes[i0] - x_nodes[i0] * y_nodes[i0+1]) / h[i0] - \
          (h[i0] * (x_nodes[i0+1] * coeffs_a[i0] - x_nodes[i0] * coeffs_a[i0+1])) / 6
          
    return term1 + term2 + bi0 * x_val + ci0

def main():
    x0, xn = 0.0, 2.0
    n_points = 10
    m_degree = 4
    x_bar = 1.5
    da, db = 0.0, 8.0

    x_nodes, y_nodes = generate_interpolation_nodes(x0, xn, n_points, target_function)
    f_x_bar = target_function(x_bar)

    ls_coeffs = solve_least_squares(x_nodes, y_nodes, m_degree)
    # Horner's method needs reversed coefficients, i.e. [am, ..., a0]
    ls_coeffs = ls_coeffs[::-1]
    
    print("--- Least Squares Results ---")
    p_m_x_bar = evaluate_polynomial_horner(ls_coeffs, x_bar)
    ls_error_sum = sum((evaluate_polynomial_horner(ls_coeffs, xi) - yi)**2 for xi, yi in zip(x_nodes, y_nodes))
    
    print(f"Pm(x_bar): {p_m_x_bar:.6f}")
    print(f"|Pm(x_bar) - f(x_bar)|: {abs(p_m_x_bar - f_x_bar):.6e}")
    print(f"LS Error Sum: {ls_error_sum:.6e}\n")
    
    print("--- Cubic Spline Results ---")
    spline_a, spline_h = solve_cubic_spline(x_nodes, y_nodes, da, db)
    s_f_x_bar = evaluate_spline(x_bar, x_nodes, y_nodes, spline_a, spline_h)

    print(f"Sf(x_bar): {s_f_x_bar:.6f}")
    print(f"|Sf(x_bar) - f(x_bar)|: {abs(s_f_x_bar - f_x_bar):.6e}")

    x_plot = np.linspace(x0, xn, 200)
    y_true = target_function(x_plot)
    y_poly = [evaluate_polynomial_horner(ls_coeffs, val) for val in x_plot]
    y_spline = [evaluate_spline(val, x_nodes, y_nodes, spline_a, spline_h) for val in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'k-', label='True Function f(x)', linewidth=2)
    plt.plot(x_plot, y_poly, 'r--', label=f'Least Squares Pm (m={m_degree})')
    plt.plot(x_plot, y_spline, 'b:', label='Cubic Spline Sf')
    plt.scatter(x_nodes, y_nodes, color='green', label='Nodes', zorder=5)
    plt.axvline(x_bar, color='orange', linestyle='-', alpha=0.3, label=f'x_bar={x_bar}')
    
    plt.title("Comparison of Function Approximations")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
