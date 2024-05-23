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
    return (135135*x-17325*(x**3) +378*(x**5)-x**7)/(135135-62370*(x**2) +3150*(x**4)-28*(x**6))
def f8(x):
    return (2027025*x - 270270*(x**3) +6930*(x**5)-36*(x**7))/(2027025-945945*(x**2) +51975*(x**4)-630*(x**6) + x**8)
def f9(x):
    return (34459425*x-4729725*(x**3) +135135*(x**5)-990*(x**7) + x**9)/(34459425-16216200*(x**2) +945945*(x**4)-13860*(x**6) +45*(x**8))


x_values = np.random.uniform(-math.pi/2, math.pi/2, 10000)

tan_values = [math.tan(x) for x in x_values]

errors = [[abs(f(x) - math.tan(x)) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]
mean_errors = [np.mean(err) for err in errors]

function_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
function_errors = list(zip(function_names, mean_errors))
function_errors.sort(key=lambda x: x[1])

sorted_function_names, sorted_errors = zip(*function_errors)


plt.figure(figsize=(10, 6))
plt.bar(sorted_function_names, sorted_errors)
plt.xlabel('Approximations')
plt.ylabel('Mean error')
plt.title('Best to worst functions')
plt.show()
f3_values = [f3(x) for x in x_values]
f4_values = [f4(x) for x in x_values]
f5_values = [f5(x) for x in x_values]
f6_values = [f6(x) for x in x_values]
f7_values = [f7(x) for x in x_values]
f8_values = [f8(x) for x in x_values]
f9_values = [f9(x) for x in x_values]


##bonus... very repetitive
def S(T, x):
    return T(x) / math.sqrt(1 + T(x)**2)
def C(T, x):
    return 1 / math.sqrt(1 + T(x)**2)

sin_approximations = [[S(f, x) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]
cos_approximations = [[C(f, x) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]

sin_errors = [[abs(S(f, x) - math.sin(x)) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]
cos_errors = [[abs(C(f, x) - math.cos(x)) for x in x_values] for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9]]

sin_mean_errors = [np.mean(err) for err in sin_errors]
cos_mean_errors = [np.mean(err) for err in cos_errors]

sin_function_errors = list(zip(function_names, sin_mean_errors))
cos_function_errors = list(zip(function_names, cos_mean_errors))

sin_function_errors.sort(key=lambda x: x[1])
cos_function_errors.sort(key=lambda x: x[1])

sorted_sin_function_names, sorted_sin_errors = zip(*sin_function_errors)
sorted_cos_function_names, sorted_cos_errors = zip(*cos_function_errors)

plt.figure(figsize=(10, 6))
plt.bar(sorted_sin_function_names, sorted_sin_errors)
plt.xlabel('Approximations')
plt.ylabel('Mean error')
plt.title('Best to worst sine functions')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(sorted_cos_function_names, sorted_cos_errors)
plt.xlabel('Approximations')
plt.ylabel('Mean error')
plt.title('Best to worst cosine functions')
plt.show()
