from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def carregar_dados_sigmoidais(arquivo: str):
    dados = np.genfromtxt(arquivo, delimiter=",")
    X = dados[:, 0:2]
    y = dados[:, 2].reshape(X.shape[0], 1)
    return (X, y)


def sinal(x):
    if x >= 0:
        return 1
    else:
        return -1


def rdc():
    return np.random.rand(3)


def embaralhar_dados(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    seed = np.random.permutation(X.shape[0])
    X_random = X[seed, :]
    y_random = y[seed, :]
    return (X_random, y_random)


def processar_dados(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    N, _ = X.shape

    (X_random, y_random) = embaralhar_dados(X, y)
    X_treino = X_random[0 : int(N * 0.8), :]
    y_treino = y_random[0 : int(N * 0.8), :]
    X_teste = X_random[int(N * 0.8) :, :]
    y_teste = y_random[int(N * 0.8) :, :]
    return (X_treino, y_treino, X_teste, y_teste, X_random, y_random)
