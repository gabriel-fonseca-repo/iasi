import numpy as np


def f_1(x, y):
    return x**2 + y**2


def f_2(x, y):
    return np.exp(-(x**2 + y**2)) + 2 * np.exp(-((x - 1.7) ** 2 + (y - 1.7) ** 2))


def f_3(x, y):
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + 20
        + np.exp(1)
    )


def f_4(x, y):
    return x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + 20


def f_5(x, y):
    return 0.5 + (
        (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5)
        / (1 + 0.001 * (x**2 + y**2)) ** 2
    )


def f_6(x, y):
    return (
        0.5
        + (
            (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5)
            / (1 + 0.001 * (x**2 + y**2)) ** 2
        )
        - 0.5
        * (
            1
            + 0.001 * (x**2 + y**2)
            - 0.5 * (np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y))
        )
    )


def f_7(x, y):
    return np.sin(x) * np.cos(y)


def f_8(x, y):
    return (
        np.sin(x)
        * np.cos(y)
        * (1 - np.exp(np.abs(100 - np.sqrt(x**2 + y**2)) / 100))
    )


LISTA_FUNCOES = [
    {
        "funcao": f_3,
        "x_lb": -8.0,
        "x_ub": 8.0,
        "max": False,
        "hillclimbing_config": {
            "max_viz": 100,
            "e": 0.1,
        },
    }
]


LISTA_FUNCOES_D = [
    {
        "funcao": f_1,
        "x_lb": -100.0,
        "x_ub": 100.0,
        "max": False,
    },
    {
        "funcao": f_2,
        "x_lb": -2.0,
        "x_ub": 5.0,
        "max": True,
    },
    {
        "funcao": f_3,
        "x_lb": -8.0,
        "x_ub": 8.0,
        "max": False,
    },
    {
        "funcao": f_4,
        "x_lb": -5.12,
        "x_ub": 5.12,
        "max": False,
    },
    {
        "funcao": f_5,
        "x_lb": -2,
        "x_ub": 3,
        "max": False,
    },
    {
        "funcao": f_6,
        "x_lb": -1,
        "x_ub": 3,
        "max": True,
    },
    {
        "funcao": f_7,
        "x_lb": 0,
        "x_ub": np.pi,
        "max": False,
    },
    {
        "funcao": f_8,
        "x_lb": -200,
        "x_ub": 20,
        "max": False,
    },
]
