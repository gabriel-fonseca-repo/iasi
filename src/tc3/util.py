from typing import Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plotar_funcao(
    xx: np.ndarray[Any, np.dtype[Any]],
    yy: np.ndarray[Any, np.dtype[Any]],
    f: callable,
) -> plt.Axes:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, f(xx, yy), alpha=0.5, cmap="jet", rstride=20, cstride=20)
    return ax


# Gera dados entre os dois limites passados
def perturb(x, e):
    return np.random.uniform(low=x + e, high=x - e)


def hillclimbing(
    x_opt: np.ndarray[Any, np.dtype[Any]],
    f: callable,  # Função a ser otimizada
    max=True,  # Se é máximo ou mínimo
    max_it=100,  # Número máximo de iterações
    e=0.1,  # Tamanho da vizinhança
    max_viz=10,  # Para cada vez que há um ótimo verifica max_viz vizinhos
    melhoria=True,  # Em quanto tem melhoria repete, caso contrário para
) -> List[np.ndarray[Any, np.dtype[Any]]]:
    i = 0
    list_prog_x_opt: List[Tuple[np.ndarray[Any, np.dtype[Any]], np.int32]] = []
    f_opt = f(x_opt[0, 0], x_opt[1, 0])
    while i < max_it and melhoria:
        melhoria = False
        i += 1
        for _ in range(max_viz):
            x_vizinho = perturb(x_opt, e)
            f_vizinho = f(x_vizinho[0, 0], x_vizinho[1, 0])

            max_or_min = f_vizinho > f_opt if max else f_vizinho < f_opt

            if max_or_min:
                x_opt = x_vizinho
                f_opt = f_vizinho
                list_prog_x_opt.append((x_opt, f_opt))
                melhoria = True
                break

    return list_prog_x_opt
