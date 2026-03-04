import math
def normalize(input: float) -> float:
    """
    Reduces the input complexity in cases it doesn't fit the (-pi/2, pi/2) interval.
    Raise IOError if the input is equal to math.pi/2.
    """
    # antisimetria tan(x) = −tan(−x).
    factor = 1.0
    if input < 0:
        input *= -1
        factor *= -1
    
    # periodicitatea functiei
    while input >= math.pi:
        input -= math.pi
    
    if abs(input - math.pi / 2) < 1e-15:
        raise IOError("Invalid floating-point number: tan(pi/2) is undefined")

    # daca input e in (pi/2, pi), folosim tan(x) = tan(x - pi) => factor se inverseaza
    if input > math.pi / 2:
        input -= math.pi
        # input e acum negativ, deci il facem pozitiv si inversam factorul
        input *= -1
        factor *= -1

    return input * factor

def tan_cont_frac (input: float) -> float:
    """Incercam sa aproximam functia tan folosind fractii continue (lentz)"""
    mic = 1e-12

    #initializare
    fct = mic
    C = fct
    D = 0.0

    x_sq = input * input

    # Prima iteratie: a1 = x, b1 = 1
    a, b = input, 1.0
    for _ in range(100_000):
        
        D = b + a * D
        if D == 0:
            D = mic
        D = 1 / D

        C = b + a / C
        if C == 0:
            C = mic
        
        diff = C * D
        fct = diff * fct

        if abs(diff - 1) < mic:
            break

        # De la j>=2: a = -x^2 (constant), b creste cu 2
        a = -x_sq
        b += 2
    
    return fct

def tan_poly_approx (input: float) -> float:
    """Incercam sa aproximam functia tan folosind polinoame"""
    x = input

    # anti simietria functiei 
    sign = 1.0
    if x < 0:
        x = -x
        sign = -1.0

    # Reducere la (-pi/4, pi/4): tan(x) = 1/tan(pi/2 - x) pt x in [pi/4, pi/2)
    use_reciprocal = False
    if x > math.pi / 4:
        x = math.pi / 2.0 - x
        use_reciprocal = True

    # Coeficienti precomputati
    c1 = 0.33333333333333333   # 1/3
    c2 = 0.13333333333333333   # 2/15
    c3 = 0.053968253968254     # 17/315
    c4 = 0.0218694885361552    # 62/2835

    x2 = x * x
    x3 = x2 * x

    # Forma Horner
    result = x + x3 * (c1 + x2 * (c2 + x2 * (c3 + x2 * c4)))

    if use_reciprocal:
        if abs(result) < 1e-15:
            result = 1e15
        else:
            result = 1.0 / result

    return sign * result

def print_to_file(filename: str, iteration: int, input_val: float,
                  result_libr: float, result_frac: float, result_poly: float):
    """Functie ajutatoare pentru a vedea incercarile functiei tangente """
    mode = "w" if iteration == 0 else "a"
    try:
        with open(filename, mode) as f:
            if iteration == 0:
                #header
                f.write(f"{'Nr':>5} | {'x':>22} | {'math.tan':>22} | {'cont_frac':>22} | {'poly_approx':>22} | {'err_frac':>10} | {'err_poly':>10} \n")
                f.write("-" * 185 + "\n")
            err_frac = abs(result_libr - result_frac)
            err_poly = abs(result_libr - result_poly)

            #results
            f.write(f"{iteration:>5} | {input_val:>22.15f} | {result_libr:>22.15f} | {result_frac:>22.15f} | {result_poly:>22.15f} | {err_frac:>10.2e} | {err_poly:>10.2e}\n")
    except IOError as e:
        print(f"Eroare la scrierea in fisier '{filename}': {e}")