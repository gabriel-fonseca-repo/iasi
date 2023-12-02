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
    ax.scatter(
        x_opt[0, 0],
        x_opt[1, 0],
        f_opt,
        s=200,
        color="r",
        marker="X",
        zorder=10000,
        edgecolors="k",
    )


def extrair_resultados(
    xx: np.ndarray,
    yy: np.ndarray,
    f: callable,
    list_prog_x_opt: List[Tuple[np.ndarray, np.int32]],
    algorithm: str,
    i: int,
):
    f_out_name = f"out/tc3/FUNCAO_{i}_VISUALIZACAO.png"
    algo_out_name = f"out/tc3/FUNCAO_{i}_CAMINHO_PERCORRIDO_{algorithm}.png"
    ax = plotar_funcao(xx, yy, f)
    plt.savefig(f_out_name, dpi=300)
    plotar_trilha(ax, list_prog_x_opt)
    plt.savefig(algo_out_name, dpi=300)
    plt.close()
    plt.clf()


def hillclimbing(
    f: callable,  # Função a ser otimizada
    max: bool,  # Se é máximo ou mínimo
    x_bound: float,  # X lower body
    y_bound: float,  # X upper body
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
    x_opt = np.array([[x_bound["lb"]], [x_bound["lb"]]])
    f_opt = f(x_opt[0, 0], x_opt[1, 0])

    while i < max_it and melhoria:
        melhoria = False
        i += 1
        for _ in range(max_viz):
            x_candidato = perturb(x_opt, e)
            f_candidato = f(x_candidato[0, 0], x_candidato[1, 0])
            max_or_min = f_candidato > f_opt if max else f_candidato < f_opt
            if max_or_min:
                x_opt = x_candidato
                f_opt = f_candidato
                list_prog_x_opt.append((x_opt, f_opt))
                melhoria = True
                break
    if not melhoria:
        list_prog_x_opt.append((x_opt, f_opt))

    return list_prog_x_opt


def tempera(
    f: callable,  # Função a ser otimizada
    max: bool,  # Se é máximo ou mínimo
    x_bound: float,  # X lower body
    y_bound: float,  # X upper body
    max_it=100,  # Número máximo de iterações
    sigma=0.01,  # Valor da variância
    t=1000,  # Temperatura inicial
) -> List[np.ndarray]:
    npru = np.random.uniform(0, 1)
    list_prog_x_opt: List[Tuple[np.ndarray, np.int32]] = []

    x_opt = np.array([[x_bound["lb"]], [x_bound["lb"]]])
    f_opt = f(x_opt[0, 0], x_opt[1, 0])

    for _ in range(max_it):
        n = np.random.normal(0, sigma)
        x_candidato = x_opt + n  # Perturbação do x ótimo

        if x_candidato[0, 0] < x_bound["lb"]:
            x_candidato[0, 0] = x_bound["lb"]
        if x_candidato[1, 0] < y_bound["lb"]:
            x_candidato[1, 0] = y_bound["lb"]

        if x_candidato[0, 0] > x_bound["ub"]:
            x_candidato[0, 0] = x_bound["ub"]
        if x_candidato[1, 0] > y_bound["ub"]:
            x_candidato[1, 0] = y_bound["ub"]

        f_candidato = f(x_candidato[0, 0], x_candidato[1, 0])
        P_ij = np.exp(-((f_candidato - f_opt) / t))

        max_or_min_f = f_candidato > f_opt if max else f_candidato < f_opt

        if max_or_min_f or P_ij >= npru:
            x_opt = x_candidato
            f_opt = f_candidato
            list_prog_x_opt.append((x_opt, f_opt))

        t = t * 0.99
    return list_prog_x_opt


def lrs(
    f: callable,  # Função a ser otimizada
    max: bool,  # Se é máximo ou mínimo
    x_bound: float,  # X lower body
    y_bound: float,  # X upper body
    max_it=100,  # Número máximo de iterações
    sigma=0.01,  # Valor da variância
) -> List[np.ndarray]:
    x_opt = np.array([[x_bound["lb"]], [x_bound["lb"]]])
    f_opt = f(x_opt[0, 0], x_opt[1, 0])

    list_prog_x_opt: List[Tuple[np.ndarray, np.int32]] = []

    for _ in range(max_it):
        n = np.random.normal(0, sigma)
        x_candidato = x_opt + n  # Perturbação do x ótimo

        if x_candidato[0, 0] < x_bound["lb"]:
            x_candidato[0, 0] = x_bound["lb"]
        if x_candidato[1, 0] < y_bound["lb"]:
            x_candidato[1, 0] = y_bound["lb"]

        if x_candidato[0, 0] > x_bound["ub"]:
            x_candidato[0, 0] = x_bound["ub"]
        if x_candidato[1, 0] > y_bound["ub"]:
            x_candidato[1, 0] = y_bound["ub"]

        f_candidato = f(x_candidato[0, 0], x_candidato[1, 0])

        max_or_min = f_candidato > f_opt if max else f_candidato < f_opt

        if max_or_min:
            x_opt = x_candidato
            f_opt = f_candidato
            list_prog_x_opt.append((x_opt, f_opt))
    return list_prog_x_opt
