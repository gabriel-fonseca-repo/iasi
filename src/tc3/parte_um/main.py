import numpy as np
from algorithms import (
    hillclimbing,
    lrs,
    grs,
    tempera,
    extrair_resultados,
    plotar_e_salvar_funcao,
)
from functions import LISTA_FUNCOES


executar_todas_funcoes = False


for i, f_dict in enumerate(LISTA_FUNCOES):
    f = f_dict["funcao"]
    max = f_dict["max"]
    x_bound = f_dict["x_bound"]
    y_bound = f_dict["y_bound"]
    current = f_dict.get("current", executar_todas_funcoes)
    indice_func = i + 1

    hiper_p_hillclimbing = f_dict["hiper_p_hillclimbing"]
    hiper_p_tempera = f_dict["hiper_p_tempera"]
    hiper_p_lrs = f_dict["hiper_p_lrs"]
    hiper_p_grs = f_dict["hiper_p_grs"]

    if not current:
        continue

    x_axis = np.linspace(x_bound["lb"], x_bound["ub"], 2000)
    y_axis = np.linspace(y_bound["lb"], y_bound["ub"], 2000)
    xx, yy = np.meshgrid(x_axis, y_axis)

    ax = plotar_e_salvar_funcao(xx, yy, f, indice_func)

    # fmt: off
    list_prog_x_hillclimbing = hillclimbing(f, max, x_bound, y_bound, **hiper_p_hillclimbing)
    extrair_resultados(xx, yy, f, ax, list_prog_x_hillclimbing, "HILLCLIMBING", indice_func)

    list_prog_x_tempera = tempera(f, max, x_bound, y_bound, **hiper_p_tempera)
    extrair_resultados(xx, yy, f, ax, list_prog_x_tempera, "TEMPERA", indice_func)

    list_prog_x_lrs = lrs(f, max, x_bound, y_bound, **hiper_p_lrs)
    extrair_resultados(xx, yy, f, ax, list_prog_x_lrs, "LRS", indice_func)

    list_prog_x_grs = grs(f, max, x_bound, y_bound, **hiper_p_grs)
    extrair_resultados(xx, yy, f, ax, list_prog_x_grs, "GRS", indice_func)
    # fmt: on
