import time
import random


UNIT = 1.0
"""Global float constant used internally."""

def find_machine_precision():
    """Returns the machine precision and the computed power (m)."""
    
    # The number of iterations = m = power.
    power = 0
    old_precision = 10
    machine_precision = 0.1
    
    while UNIT + machine_precision > UNIT:
        old_precision = machine_precision
        machine_precision /= 10
        power += 1

    # UNIT + old_precision / 10 == UNIT 
    return old_precision, power - 1

if __name__ == "__main__":
    machine_precision, power = find_machine_precision()
    print(f"Machine precision: {machine_precision}, m: {power}")
    print(f"First verification (should NOT be 1.0): {UNIT + machine_precision}")
    print(f"Second verification (should be 1.0): {UNIT + machine_precision / 10}")


# hmm interesant if __name__== "__main__" :D 


#EXERCITIUL 2
u, _ = find_machine_precision()
x = 1.0
y = u / 10
z = u / 10

st = (x + y) + z    # (1.0 + u/10) + u/10
dr = x + (y + z)    # 1.0 + (u/10 + u/10)

if st!=dr:
    print(f"(x + y) + z = {st}")
    print(f"x + (y + z) = {dr}")
    print(f"Neasociativ  {st!=dr}")
# acum trebuie sa gasim x y z pentru care inmultirea ESTE ASOCIATIVA

s=time.time()
for incercare in range(1,100_001):
    x = 10 ** random.uniform(40, 200)
    y = 10 ** random.uniform(-180, -100)
    z = 10 ** random.uniform(-155, -90)
    st = (x * y) * z
    dr = x * (y * z)

    if st != dr and st > 0 and dr == 0.0:
        print(f"Gasit dupa {incercare} incercari \n x = {x:.2e}, y = {y:.2e}, z = {z:.2e}")
        print(f"(x*y)*z = {st:.2e}")
        print(f"x*(y*z) = {dr}")
        break

print(f"Timp total {time.time()-s:.4f} secunde")

     

