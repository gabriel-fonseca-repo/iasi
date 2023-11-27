# Algoritmo de busca aleatória local
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * np.sin(10 * x * np.pi) + 1


x_lb = -1  # x lower body
x_ub = 2  # x upper body

x_axis = np.linspace(x_lb, x_ub, 1000)

# hiper-parametros
it_max = 1000  # Número máximo de iterações
sigma = 0.01  # Acho que é o valor da variância
x_opt = np.random.uniform(low=x_lb, high=x_ub)  # x ótimo
f_opt = f(x_opt)

for i in range(it_max):
    n = np.random.normal(0, sigma)
    x_candidato = x_opt + n  # Perturbação do x ótimo

    if x_candidato > x_ub:
        x_candidato = x_ub
    if x_candidato < x_lb:
        x_candidato = x_lb

    f_candidato = f(x_candidato)
    if f_candidato > f_opt:
        x_opt = x_candidato
        f_opt = f_candidato

plt.grid(True)
plt.xlim(-1.1, 2.1)
plt.plot(x_axis, f(x_axis))
plt.show()
