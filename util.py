import numpy as np
from pyparsing import Any


def estimar_modelo(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return (np.linalg.inv(X.T @ X) @ X.T @ y, X)


def processar_dados(X: np.ndarray[Any, np.dtype[Any]], y):
    N, p = X.shape

    seed = np.random.permutation(N)
    X_random = X[seed, :]
    y_random = y[seed, :]
    X_treino = X_random[0 : int(N * 0.8), :]
    y_treino = y_random[0 : int(N * 0.8), :]
    X_teste = X_random[int(N * 0.8) :, :]
    y_teste = y_random[int(N * 0.8) :, :]
    return (X_treino, y_treino, X_teste, y_teste, X_random, y_random)
