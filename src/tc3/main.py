import numpy as np
from algorithms import hillclimbing, plotar_funcao, plotar_trilha
from functions import LISTA_FUNCOES


for f_dict in LISTA_FUNCOES:
    f = f_dict["funcao"]
    x_lb = f_dict["x_lb"]
    x_ub = f_dict["x_ub"]
    max = f_dict["max"]

    if f is None:
        continue

    x_axis = np.linspace(x_lb, x_ub, 1000)
    xx, yy = np.meshgrid(x_axis, x_axis)

    ax = plotar_funcao(xx, yy, f)
    list_prog_x_opt = hillclimbing(f, x_lb, x_ub, max=max)
    plotar_trilha(ax, list_prog_x_opt)
