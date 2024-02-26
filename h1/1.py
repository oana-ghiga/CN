#1
m = 1
u = 10**(-m)
while 1+u == 1:
    m += 1
    u = 10**(-m)
print(u)
print(m)
# the smallest number u that satisfies the propriety 1(+c) u != 1 is 1e-16
