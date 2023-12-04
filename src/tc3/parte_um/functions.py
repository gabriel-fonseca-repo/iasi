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
