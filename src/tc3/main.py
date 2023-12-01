import numpy as np
import matplotlib.pyplot as plt
from util import hillclimbing, plotar_funcao


def f_1(x, y):
    return x**2 + y**2


def f_2(x, y):
    return np.exp(-(x**2 + y**2))


def f_3(x, y):
    return np.exp(-(x**2 + y**2)) + 2 * np.exp(-((x - 1.7) ** 2 + (y - 1.7) ** 2))


# Gerando os dados para plotar a função
x_axis = np.linspace(-2, 2, 1000)  # Usar para f(x,y)
x_axis_2 = np.linspace(-2, 4, 1000)  # Usar para f2(x,y)
xx, yy = np.meshgrid(x_axis, x_axis)

ax = plotar_funcao(xx, yy, f_1)

# X ótimo, que pode ser inicializado aleatoriamente.
# Aqui foi inicializado como o limite inferior das funções.
x_opt = np.array([[-2], [-2]])

# Executando o algoritmo de hillclimbing
list_prog_x_opt = hillclimbing(x_opt, f_1, max=False)

for x_opt, f_opt in list_prog_x_opt:
    ax.scatter(x_opt[0, 0], x_opt[1, 0], f_opt, s=40, color="k")
    plt.pause(0.001)
