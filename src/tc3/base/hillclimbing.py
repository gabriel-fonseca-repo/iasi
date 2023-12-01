import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return np.exp(-(x**2 + y**2))


def f2(x, y):
    return np.exp(-(x**2 + y**2)) + 2 * np.exp(-((1.7 - x) ** 2 + (1.7 - y) ** 2))


x_axis = np.linspace(-2, 2, 1000)  # Usar para f(x,y)
x_axis2 = np.linspace(-2, 4, 1000)  # Usar para f2(x,y)
xx, yy = np.meshgrid(x_axis, x_axis)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx, yy, f(xx, yy), alpha=0.5, cmap="jet", rstride=20, cstride=20)

# hiperparametros
max_it = 100  # iterações máximas
e = 0.1  # tamanho da vizinhança
max_viz = 10  # Para cada vez que há um ótimo verifica max_viz vizinhos
melhoria = True  # Em quanto tem melhoria repete, caso contrário para

# x ótimo, que pode ser inicializado aleatoriamente, mas aqui foi inicializado
# como um dos limites da função
x_opt = np.array([[-2], [-2]])
f_opt = f(x_opt[0, 0], x_opt[1, 0])

# Aqui é apenas uma visualização do ponto inicial
# ax.scatter(x_opt[0,0], x_opt[1,0], f_opt, s=40, color='k')


# Gera dados dentre o limite da vizinhança
def perturb(x, e):
    return np.random.uniform(low=x - e, high=x + e)


i = 0
while i < max_it and melhoria:
    melhoria = False
    i += 1
    for j in range(max_viz):
        # Gera um vizinho a ser avaliado
        x_vizinho = perturb(x_opt, e)
        f_vizinho = f(x_vizinho[0, 0], x_vizinho[1, 0])
        if f_vizinho > f_opt:
            x_opt = x_vizinho
            f_opt = f_vizinho
            melhoria = True
            # Visualização do novo ponto ótimo
            ax.scatter(x_opt[0, 0], x_opt[1, 0], f_opt, s=40, color="k")
            plt.pause(0.05)
            break
plt.show()
