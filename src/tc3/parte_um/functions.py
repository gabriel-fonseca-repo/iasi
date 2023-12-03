import numpy as np


def f_1(x, y):
    return x**2 + y**2


def f_2(x, y):
    return np.exp(-(x**2 + y**2)) + 2 * np.exp(-((x - 1.7) ** 2 + (y - 1.7) ** 2))


def f_3(x, y):
    return (
        -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2))
        - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
        + 20
        + np.exp(1)
    )


def f_4(x, y):
    return (x**2 - 10 * np.cos(2 * np.pi * x) + 10) + (
        y**2 - 10 * np.cos(2 * np.pi * y) + 10
    )


def f_5(x, y):
    return (x - 1) ** 2 + 100 * (y - x**2) ** 2


def f_6(x, y):
    return (x * np.sin(4 * np.pi * x)) - (y * np.sin(4 * np.pi * y + np.pi)) + 1


def f_7(x, y):
    return (
        -np.sin(x) * np.sin(x**2 / np.pi) ** 20
        - np.sin(y) * np.sin(2 * y**2 / np.pi) ** 20
    )


def f_8(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + y + 47))) - x * np.sin(
        np.sqrt(np.abs(x - y - 47))
    )


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
            "max_it": 100,
            "sigma": 5.0,
        },
        "hiper_p_grs": {
            "max_it": 1000,
        },
        "hiper_p_tempera": {
            "max_it": 1000,
            "sigma": 7.0,
            "t": 10,
        },
    },
    # Função 2 do TC3
    {
        "current": True,
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
            "max_it": 100,
            "sigma": 0.8,
            "t": 1000,
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
            "sigma": 0.01,
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
            "sigma": 0.01,
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
            "max_it": 100,
            "sigma": 0.01,
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
    # Função 6 do TC3
    {
        "funcao": f_6,
        "max": False,
        "x_bound": {
            "lb": -1.0,
            "ub": 3.0,
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
            "max_it": 100,
            "sigma": 0.01,
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
            "max_viz": 300,
            "max_it": 100,
            "e": 0.1,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.01,
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
            "max_viz": 300,
            "max_it": 100,
            "e": 0.1,
        },
        "hiper_p_lrs": {
            "max_it": 100,
            "sigma": 0.01,
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
