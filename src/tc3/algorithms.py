from typing import Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plotar_funcao(
    xx: np.ndarray,
    yy: np.ndarray,
    f: callable,
) -> plt.Axes:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, f(xx, yy), alpha=0.5, cmap="jet", rstride=20, cstride=20)
    return ax


def plotar_trilha(ax: plt.Axes, list_prog_x_opt: List[Tuple[np.ndarray, np.int32]]):
    for x_opt, f_opt in list_prog_x_opt:
        ax.scatter(x_opt[0, 0], x_opt[1, 0], f_opt, s=40, color="k")
        plt.pause(0.001)
    ax.scatter(x_opt[0, 0], x_opt[1, 0], f_opt, s=40, color="r", marker="X")
    plt.show()


def hillclimbing(
    f: callable,  # Função a ser otimizada
    x_lb: float,  # X lower body
    x_ub: float,  # X upper body
    max=True,  # Se é máximo ou mínimo
    max_it=100,  # Número máximo de iterações
    e=0.1,  # Tamanho da vizinhança
    max_viz=10,  # Para cada vez que há um ótimo verifica max_viz vizinhos
) -> List[np.ndarray]:
    i = 0
    melhoria = (True,)  # Em quanto tem melhoria repete, caso contrário para

    perturb = lambda x, e: np.random.uniform(low=x + e, high=x - e)

    list_prog_x_opt: List[Tuple[np.ndarray, np.int32]] = []

    # X ótimo, que pode ser inicializado aleatoriamente.
    # Aqui foi inicializado como o limite inferior das funções.
    x_opt = np.array([[x_lb], [x_lb]])
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
    if not melhoria:
        list_prog_x_opt.append((x_opt, f_opt))

    return list_prog_x_opt


def tempera(
    f: callable,  # Função a ser otimizada
    x_lb: float,  # X lower body
    x_ub: float,  # X upper body
    max=True,  # Se é máximo ou mínimo
    max_it=100,  # Número máximo de iterações
    sigma=0.01,  # Valor da variância
):
    npru = np.random.uniform(0, 1)

    for _ in range(max_it):
        n = np.random.normal(0, sigma)
        x_candidato = x_opt + n  # Perturbação do x ótimo

        if x_candidato > x_ub:
            x_candidato = x_ub
        if x_candidato < x_lb:
            x_candidato = x_lb

        f_candidato = f(x_candidato)
        P_ij = np.exp(-((f_candidato - f_opt) / T))

        max_or_min_f = f_candidato > f_opt if max else f_candidato < f_opt

        if max_or_min_f or P_ij >= npru:
            x_opt = x_candidato
            f_opt = f_candidato

        T = T * 0.99


def lrs(
    f: callable,  # Função a ser otimizada
    max=True,  # Se é máximo ou mínimo
    max_it=100,  # Número máximo de iterações
    sigma=0.01,  # Valor da variância
):
    x_lb = -1  # X lower body
    x_ub = 2  # X upper body

    x_opt = np.random.uniform(low=x_lb, high=x_ub)  # x ótimo
    f_opt = f(x_opt)

    list_prog_x_opt: List[Tuple[np.ndarray, np.int32]] = []

    for _ in range(max_it):
        n = np.random.normal(0, sigma)
        x_candidato = x_opt + n  # Perturbação do x ótimo

        if x_candidato > x_ub:
            x_candidato = x_ub
        if x_candidato < x_lb:
            x_candidato = x_lb

        f_candidato = f(x_candidato)

        max_or_min = f_candidato > f_opt if max else f_candidato < f_opt

        if max_or_min:
            x_opt = x_candidato
            f_opt = f_candidato
            list_prog_x_opt.append((x_opt, f_opt))
    return list_prog_x_opt
