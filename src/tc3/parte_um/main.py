import numpy as np
from algorithms import (
    hillclimbing,
    lrs,
    grs,
    tempera,
    extrair_resultados,
    plotar_e_salvar_funcao,
)
from functions import f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8

LISTA_FUNCOES = [
    # Função 1 do TC3
    {
        "funcao": f_1,
        "max": False,
        "x_bound": {
            "lb": -100.0,
            "ub": 100.0,
        },
        "y_bound": {
            "lb": -100.0,
            "ub": 100.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 200,
            "max_it": 100,
            "e": 5,
        },
        "hiper_p_lrs": {
            "max_it": 1000,
            "sigma": 0.8,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 1000,
            "sigma": 0.9,
            "t": 1000,
        },
    },
    # Função 2 do TC3
    {
        "funcao": f_2,
        "max": True,
        "x_bound": {
            "lb": -2.0,
            "ub": 4.0,
        },
        "y_bound": {
            "lb": -2.0,
            "ub": 5.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 500,
            "max_it": 100,
            "e": 1.2,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.8,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 1000,
            "sigma": 1.8,
            "t": 10,
        },
    },
    # Função 3 do TC3
    {
        "funcao": f_3,
        "max": False,
        "x_bound": {
            "lb": -8.0,
            "ub": 8.0,
        },
        "y_bound": {
            "lb": -8.0,
            "ub": 8.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 300,
            "max_it": 100,
            "e": 2.0,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.8,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
    # Função 4 do TC3
    {
        "funcao": f_4,
        "max": False,
        "x_bound": {
            "lb": -5.12,
            "ub": 5.12,
        },
        "y_bound": {
            "lb": -5.12,
            "ub": 5.12,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 300,
            "max_it": 100,
            "e": 2.0,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.7,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
    # Função 5 do TC3
    {
        "funcao": f_5,
        "max": False,
        "x_bound": {
            "lb": -2.0,
            "ub": 2.0,
        },
        "y_bound": {
            "lb": -1.0,
            "ub": 3.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 300,
            "max_it": 100,
            "e": 0.1,
        },
        "hiper_p_lrs": {
            "max_it": 1000,
            "sigma": 0.7,
        },
        "hiper_p_grs": {
            "max_it": 10000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
    # Função 6 do TC3
    {
        "funcao": f_6,
        "max": True,
        "x_bound": {
            "lb": -1.0,
            "ub": 3.0,
        },
        "y_bound": {
            "lb": -1.0,
            "ub": 3.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 400,
            "max_it": 1000,
            "e": 5.0,
        },
        "hiper_p_lrs": {
            "max_it": 1000,
            "sigma": 0.8,
        },
        "hiper_p_grs": {
            "max_it": 10000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
    # Função 7 do TC3
    {
        "funcao": f_7,
        "max": False,
        "x_bound": {
            "lb": 0,
            "ub": np.pi,
        },
        "y_bound": {
            "lb": 0,
            "ub": np.pi,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 400,
            "max_it": 100,
            "e": 2.0,
        },
        "hiper_p_lrs": {
            "max_it": 1000,
            "sigma": 0.9,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
    # Função 8 do TC3
    {
        "current": True,
        "funcao": f_8,
        "max": False,
        "x_bound": {
            "lb": -200.0,
            "ub": 20.0,
        },
        "y_bound": {
            "lb": -200.0,
            "ub": 20.0,
        },
        "hiper_p_hillclimbing": {
            "max_viz": 500,
            "max_it": 1000,
            "e": 70,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.2,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 100,
            "sigma": 0.01,
            "t": 1000,
        },
    },
]


executar_todas_funcoes = True
printar_graficos = True
medir_moda = False

if printar_graficos:
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


MODA_RST_OPT = {
    "1": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "2": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "3": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "4": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "5": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "6": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "7": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
    "8": {
        "HILLCLIMBING": [],
        "TEMPERA": [],
        "LRS": [],
        "GRS": [],
    },
}

if medir_moda:
    for i in range(100):
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

            # fmt: off
            list_prog_x_hillclimbing = hillclimbing(f, max, x_bound, y_bound, **hiper_p_hillclimbing)
            list_prog_x_tempera = tempera(f, max, x_bound, y_bound, **hiper_p_tempera)
            list_prog_x_lrs = lrs(f, max, x_bound, y_bound, **hiper_p_lrs)
            list_prog_x_grs = grs(f, max, x_bound, y_bound, **hiper_p_grs)

            x_opt_hillclimbing = list_prog_x_hillclimbing[-1]
            x_opt_tempera = list_prog_x_tempera[-1]
            x_opt_lrs = list_prog_x_lrs[-1]
            x_opt_grs = list_prog_x_grs[-1]

            MODA_RST_OPT[str(indice_func)]["HILLCLIMBING"].append(x_opt_hillclimbing)
            MODA_RST_OPT[str(indice_func)]["TEMPERA"].append(x_opt_tempera)
            MODA_RST_OPT[str(indice_func)]["LRS"].append(x_opt_lrs)
            MODA_RST_OPT[str(indice_func)]["GRS"].append(x_opt_grs)
            # fmt: on

        pass
