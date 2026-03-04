def get_next_a(a: float) -> float:
    """Return the next value of a for the continued fraction method."""
    return -(a ** 2)

def get_next_b(b: float) -> float:
    """Return the next value of b for the continued fraction method."""
    return b + 2

def tan_cont_frac (input: float) -> float:
    """Return an approximation of tan(input) using continuous functions, i.e. first method (Lentz)."""
    a, b = 0.0, 0.0
    err = 1e-12
    fct = err if b == 0 else b
    C, D = fct, 0.0
    
    a, b = input, 1.0
    for _ in range(100_000):
        
        D = b + a * D
        if D == 0:
            D = err
        D = 1 / D

        C = b + a / C
        if C == 0:
            C = err
        
        diff = C * D
        fct = diff * fct
        a = get_next_a(a)
        b = get_next_b(b)
        
        if abs(diff - 1) < err:
            break
    
    return fct

def tan_poly_approx (input: float) -> float:
    """Return an approximation of tan(input) using a polynomial expansion, i.e. second method."""
    raise NotImplementedError("Exercitiul 3: tan_poly_approx nu a fost implementata")
    return input + (input ** 3) / 3 + (2 * input ** 5) / 15 + (17 * input ** 7) / 35
