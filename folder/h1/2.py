def precision():
    m = 1
    u = 10**(-m)
    while 1+u != 1:
        prev_m = m
        m += 1
        prev_u = u
        u = 10**(-m)
    return prev_u, prev_m

u,m = precision()

x = 1.0
y = u/10
z = u/10
if (x + y) + z != x + (y + z):
    print(f"Operatia de adunare efectuata de calculator nu e asociativa pentru precizia u = {u}, m = {m}")

x = 0.0001
y = 0.0001
z = 10.0

if (x * y) * z != x * (y * z):
    print(f"Operatia de inmultire efectuata de calculator nu e asociativa pentru x = {x}, y = {y}, z = {z}")