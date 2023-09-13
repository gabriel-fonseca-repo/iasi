import numpy as np
from pyparsing import Any


def estimar_modelo_ones(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.linalg.inv(X.T @ X) @ X.T @ y


def estimar_modelo_zeros(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    r = np.linalg.inv(X.T @ X) @ X.T @ y
    zero = np.zeros((1, 1))
    r = np.concatenate((zero, r), axis=1)
    return r.T


def estimar_modelo_tikhonov(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]], lbd
):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.linalg.inv(X.T @ X + lbd * np.eye(X.shape[1])) @ X.T @ y


def definir_melhor_lambda(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    X_t: np.ndarray[Any, np.dtype[Any]],
    y_t: np.ndarray[Any, np.dtype[Any]],
    lbd: list
):
    melhor_modelo = estimar_modelo_tikhonov(X, y, lbd[0])
    melhor_lambda = lbd[0]
    X_t = concatenar_uns(X_t)
    media_lambdas = []
    for x in lbd:
        media_lambdas_x = []
        for i in range(1000):
            modelo_da_vez = estimar_modelo_tikhonov(X, y, x)
            y_pred = X_t @ modelo_da_vez
            media_lambdas_x.append(eqm(y_pred, y_t))
        media_lambdas.append(np.mean(media_lambdas_x))
    melhor_lambda = np.argmin(media_lambdas)
    return melhor_lambda


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
