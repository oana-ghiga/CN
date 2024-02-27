import math
import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return x
def f2(x):
    return (3*x)/(3-x**2)
def f3(x):
    return (15*x-x**3)/(15-6*(x**2))
def f4(x):
    return (105*x-10*(x**3))/(105-45*(x**2)+x**4)
def f5(x):
    return (945*x-105*(x**3)+x**5)/(945-420*(x**2)+15*(x**4))
def f6(x):
    return (10395*x-1260*(x**3)+21*(x**5))/(10395-4725*(x**2)+210*(x**4)-x**6)
def f7(x):
    return (135135*x-18900*(x**3)+315*(x**5)-x**7)/(135135-62370*(x**2)+3150*(x**4)-84*(x**6)+x**8)
def f8(x):
    return (2027025*x-315315*(x**3)+6930*(x**5)-126*(x**7)+x**9)/(2027025-969969*(x**2)+45045*(x**4)-1980*(x**6)+66*(x**8)-x**10)
def f9(x):
    return (34459425*x-5819815*(x**3)+135135*(x**5)-2772*(x**7)+36*(x**9))/(34459425-17383860*(x**2)+850668*(x**4)-41580*(x**6)+1716*(x**8)-55*(x**10)+x**11)


x_values = np.random.uniform(-math.pi/2, math.pi/2, 10000)

f1_values = [f1(x) for x in x_values]
f2_values = [f2(x) for x in x_values]
f3_values = [f3(x) for x in x_values]
f4_values = [f4(x) for x in x_values]
f5_values = [f5(x) for x in x_values]
f6_values = [f6(x) for x in x_values]
f7_values = [f7(x) for x in x_values]
f8_values = [f8(x) for x in x_values]
f9_values = [f9(x) for x in x_values]

tan_values = [math.tan(x) for x in x_values]


errors = [[abs(f(x) - math.tan(x)) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]


top_three = [sorted(range(9), key=lambda i: errors[i][j])[:3] for j in range(10000)]


counts = [sum(i in top for top in top_three) for i in range(9)]


hierarchy = sorted(range(9), key=lambda i: -counts[i])


print("Hierarchy of functions from best to worst:")
for i in hierarchy:
    print(f"f{i+1}")


plt.bar(range(9), [counts[i] for i in hierarchy])
plt.xticks(range(9), [f"f{i+1}" for i in hierarchy])
plt.xlabel('Function')
plt.ylabel('Count')
plt.title('Hierarchy of functions')
plt.show()