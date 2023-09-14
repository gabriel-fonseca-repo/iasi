import numpy as np
from typing import Any


def mqo(X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]):
    X = concatenar_uns(X)
    return np.linalg.inv(X.T @ X) @ X.T @ y


def mqo_sem_intercept_bidimensional(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    r = np.linalg.inv(X.T @ X) @ X.T @ y
    zero = np.zeros((1, 1))
    r = np.concatenate((zero, r), axis=1)
    return r.T


def mqo_sem_intercept_tridimensional(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    r = np.linalg.inv(X.T @ X) @ X.T @ y
    zero = np.zeros((1, 1))
    r = np.concatenate((zero, r.T), axis=1)
    return r.T


def mqo_tikhonov(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]], lbd
):
    X = concatenar_uns(X)
    return np.linalg.inv(X.T @ X + lbd * np.eye(X.shape[1])) @ X.T @ y


def media_b(
    y: np.ndarray[Any, np.dtype[Any]],
):
    b_media = np.mean(y)
    b_media = np.array([[b_media], [0]])
    return b_media


def media_b_tridimensional(
    y: np.ndarray[Any, np.dtype[Any]],
):
    b_media = np.mean(y)
    b_media = np.array([[b_media], [0], [0]])
    return b_media


def knn(X: np.ndarray[Any, np.dtype[Any]], X_t: np.ndarray[Any, np.dtype[Any]], k=7):
    DISTANCIAS = []
    for i in range(X_t.shape[0]):
        X_i = X_t[i, :]
        for j in range(X.shape[0]):
            X_j = X[j, :]
            DISTANCIAS.append(np.linalg.norm(X_i - X_j))
        k_menores_indices = np.argsort(DISTANCIAS)[:k]
        print()


def dmc(X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]):
    pass


def eqm(y: np.ndarray[Any, np.dtype[Any]], y_teste: np.ndarray[Any, np.dtype[Any]]):
    return np.mean((y - y_teste) ** 2)


def eqm_classificacao_ols(
    y_teste: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    disc_teste = np.argmax(y_teste, axis=1)
    disc_hat = np.argmax(y, axis=1)
    teste_size = disc_hat.shape[0]
    count_acertos_c = np.count_nonzero(disc_teste == disc_hat) / teste_size
    return count_acertos_c


def concatenar_uns(X: np.ndarray[Any, np.dtype[Any]]):
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
