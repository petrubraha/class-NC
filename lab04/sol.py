import numpy as np
import sys
import os

EPSILON = 1e-8   # precizia default
K_MAX   = 10000  # cat de mult lasam GS sa itereze
DIV_THR = 1e10   # daca delta depaseste asta, a divergat clar

def load_vector(path: str) -> np.ndarray:
    # citeste vectorul din fisier, un numar pe linie
    return np.loadtxt(path)

def load_system(folder: str, idx: int):
    # incarca cele 4 fisiere ale sistemului i
    # d0 = diagonala principala, d1 = diag ordin p, d2 = diag ordin q, b = termeni liberi
    d0 = load_vector(os.path.join(folder, f"d0_{idx}.txt"))
    d1 = load_vector(os.path.join(folder, f"d1_{idx}.txt"))
    d2 = load_vector(os.path.join(folder, f"d2_{idx}.txt"))
    b  = load_vector(os.path.join(folder, f"b_{idx}.txt"))
    return d0, d1, d2, b

# punctul 1
def get_n(d0: np.ndarray, b: np.ndarray) -> int:
    # n = cate elemente are diagonala principala (= cat are b)
    assert len(d0) == len(b), "d0 si b au dimensiuni diferite!"
    return len(d0)

# punctul 2
def get_p_q(n: int, d1: np.ndarray, d2: np.ndarray):
    # diagonala de ordin p are n-p elemente => p = n - |d1|
    # diagonala de ordin q are n-q elemente => q = n - |d2|
    p = n - len(d1)
    q = n - len(d2)
    return p, q

# punctul 3
def check_diagonal(d0: np.ndarray, eps: float) -> bool:
    # verificam ca |a[i,i]| > eps pentru orice i
    # daca nu, GS nu poate fi aplicat
    return np.all(np.abs(d0) > eps)

# punctul 4
def gauss_seidel(d0, d1, d2, b, p, q, eps):
    # metoda Gauss-Seidel pe matricea rara stocata in d0, d1, d2
    #
    # pentru fiecare componenta i, formula e:
    #   x_new[i] = ( b[i]
    #              - d1[i-p] * x_new[i-p]   <- stanga in d1, deja actualizat
    #              - d1[i]   * x_old[i+p]   <- dreapta in d1, inca vechi
    #              - d2[i-q] * x_new[i-q]   <- stanga in d2, deja actualizat
    #              - d2[i]   * x_old[i+q]   <- dreapta in d2, inca vechi
    #              ) / d0[i]
    n  = len(d0)
    xp = np.zeros(n)   # x^(k)   — iteratia veche
    xc = np.zeros(n)   # x^(k+1) — iteratia noua

    for k in range(K_MAX):
        xp[:] = xc   # salvam xc ca sa avem x vechi la iteratia asta

        for i in range(n):
            s = b[i]   # pornim cu b[i] si scadem contributiile vecinilor

            # contributia diagonalei d1 (ordin p)
            if i >= p:        # exista vecin la stanga pe d1?  a[i, i-p] = d1[i-p]
                s -= d1[i - p] * xc[i - p]   # deja actualizat, folosim xc
            if i + p < n:     # exista vecin la dreapta pe d1? a[i, i+p] = d1[i]
                s -= d1[i] * xp[i + p]        # inca vechi, folosim xp

            # contributia diagonalei d2 (ordin q)
            if i >= q:        # exista vecin la stanga pe d2?  a[i, i-q] = d2[i-q]
                s -= d2[i - q] * xc[i - q]   # deja actualizat, folosim xc
            if i + q < n:     # exista vecin la dreapta pe d2? a[i, i+q] = d2[i]
                s -= d2[i] * xp[i + q]        # inca vechi, folosim xp

            xc[i] = s / d0[i]   # impartim la diagonala principala

        # cat de mult s-a miscat solutia fata de iteratia anterioara
        delta = np.linalg.norm(xc - xp)

        if delta < eps:   # am convergit
            print(f"  Convergenta la iteratia k={k+1}, delta={delta:.2e}")
            return xc

        if delta > DIV_THR:   # a explodat, nu are sens sa continuam
            print(f"  Divergenta la iteratia k={k+1}, delta={delta:.2e}")
            return None

    print(f"  Nu a convergat in {K_MAX} iteratii, delta={delta:.2e}")
    return None

# punctul 5
def matvec(d0, d1, d2, x, p, q):
    # calculeaza y = A * x fara sa construim A explicit
    # folosim doar d0, d1, d2 — fiecare element din d1, d2 accesat exact de 2 ori
    n = len(d0)
    y = np.zeros(n)

    for i in range(n):
        y[i] = d0[i] * x[i]   # contributia diagonalei principale

        if i >= p:       # a[i, i-p] = d1[i-p]
            y[i] += d1[i - p] * x[i - p]
        if i + p < n:    # a[i, i+p] = d1[i]
            y[i] += d1[i] * x[i + p]

        if i >= q:       # a[i, i-q] = d2[i-q]
            y[i] += d2[i - q] * x[i - q]
        if i + q < n:    # a[i, i+q] = d2[i]
            y[i] += d2[i] * x[i + q]

    return y

# punctul 6
def inf_norm(v: np.ndarray) -> float:
    # ||v||inf = max |v[i]| — cea mai mare componenta ca valoare absoluta
    return np.max(np.abs(v))


def solve_system(folder: str, idx: int, eps: float):
    print(f"\n{'='*50}")
    print(f"Sistem {idx}  (eps={eps:.0e})")
    print(f"{'='*50}")

    d0, d1, d2, b = load_system(folder, idx)

    # punctul 1 — dimensiunea
    n = get_n(d0, b)
    print(f"[1] n = {n}")

    # punctul 2 — ordinele diagonalelor
    p, q = get_p_q(n, d1, d2)
    print(f"[2] p = {p}, q = {q}")

    # punctul 3 — verificam ca diagonala e nenula
    if not check_diagonal(d0, eps):
        print("[3] EROARE: diagonala principala contine elemente nule!")
        return
    print(f"[3] Diagonala principala ok")

    # punctul 4 — Gauss-Seidel
    print(f"[4] Rulam Gauss-Seidel...")
    x_gs = gauss_seidel(d0, d1, d2, b, p, q, eps)

    if x_gs is None:
        print("[4] Nu s-a putut aproxima solutia (divergenta).")
        return

    print(f"    Primele 5 componente: {x_gs[:5]}")

    # punctul 5 — calculam y = A * xGS
    y = matvec(d0, d1, d2, x_gs, p, q)

    # punctul 6 — norma reziduului
    residual = inf_norm(y - b)
    print(f"[6] ||A*xGS - b||inf = {residual:.6e}")


def main():
    # folderul cu datele, default "data"
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"

    # precizia, default 1e-8
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else EPSILON

    for idx in range(1, 6):
        b_path = os.path.join(folder, f"b_{idx}.txt")
        if not os.path.exists(b_path):
            print(f"\nSistem {idx}: lipseste b_{idx}.txt, skip.")
            continue
        solve_system(folder, idx, eps)


if __name__ == "__main__":
    main()