import numpy as np
from algorithms import (
    hillclimbing,
    lrs,
    tempera,
    extrair_resultados,
)
from functions import LISTA_FUNCOES


for i, f_dict in enumerate(LISTA_FUNCOES):
    f = f_dict["funcao"]
    max = f_dict["max"]
    x_bound = f_dict["x_bound"]
    y_bound = f_dict["y_bound"]
    indice_func = i + 1

    hiper_p_hillclimbing = f_dict["hiper_p_hillclimbing"]
    hiper_p_tempera = f_dict["hiper_p_tempera"]
    hiper_p_lrs = f_dict["hiper_p_lrs"]

    if f is None:
        continue

    x_axis = np.linspace(x_bound["lb"], x_bound["ub"], 1000)
    xx, yy = np.meshgrid(x_axis, x_axis)

    # fmt: off
    list_prog_x_hillclimbing = hillclimbing(f, max, x_bound, y_bound, **hiper_p_hillclimbing)
    extrair_resultados(xx, yy, f, list_prog_x_hillclimbing, "HILLCLIMBING", indice_func)

    list_prog_x_tempera = tempera(f, max, x_bound, y_bound, **hiper_p_tempera)
    extrair_resultados(xx, yy, f, list_prog_x_tempera, "TEMPERA", indice_func)

    list_prog_x_lrs = lrs(f, max, x_bound, y_bound, **hiper_p_lrs)
    extrair_resultados(xx, yy, f, list_prog_x_lrs, "LRS", indice_func)
    # fmt: on
