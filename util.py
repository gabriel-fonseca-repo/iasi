import numpy as np
from pyparsing import Any


def estimar_modelo_ones(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return (np.linalg.inv(X.T @ X) @ X.T @ y, X)


def estimar_modelo_zeros(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    r = (np.linalg.inv(X.T @ X) @ X.T @ y, X)
    zero = np.zeros((1, 1))
    r = np.concatenate((zero, r), axis=0)
    return r


def estimar_modelo_tikhonov(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]], lbd
):
    return (np.linalg.inv(X.T @ X + lbd * np.eye(X.shape[1])) @ X.T @ y, X)


def concatenar_uns(X: np.ndarray[Any, np.dtype[Any]]):
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def processar_dados(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    N, p = X.shape

    seed = np.random.permutation(N)
    X_random = X[seed, :]
    y_random = y[seed, :]
    X_treino = X_random[0 : int(N * 0.8), :]
    y_treino = y_random[0 : int(N * 0.8), :]
    X_teste = X_random[int(N * 0.8) :, :]
    y_teste = y_random[int(N * 0.8) :, :]
    return (X_treino, y_treino, X_teste, y_teste, X_random, y_random)


def media_b(
    y: np.ndarray[Any, np.dtype[Any]],
):
    b_media = np.mean(y)
    b_media = np.array([[b_media], [0]])
    return b_media


def eqm(y: np.ndarray[Any, np.dtype[Any]], modelo: np.ndarray[Any, np.dtype[Any]]):
    return np.mean((y - modelo) ** 2)
