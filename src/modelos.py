import numpy as np
from typing import Any, List


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


def knn(
    X_treino: np.ndarray[Any, np.dtype[Any]],
    y_treino: np.ndarray[Any, np.dtype[Any]],
    X_teste: np.ndarray[Any, np.dtype[Any]],
    y_teste: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
    k=7,
):
    contador_acertos = 0
    for i in range(X_teste.shape[0]):
        X_i = X_teste[i, :]
        dists = []
        for j in range(X_treino.shape[0]):
            X_j = X_treino[j, :]
            dists.append(np.linalg.norm(X_i - X_j))
        k_menores_indices = np.argsort(dists)[0:k]
        k_menores_vizinhos = y_treino[k_menores_indices, :]
        rotulo_da_vez = np.argmax(k_menores_vizinhos, axis=1)
        contador_rotulos = np.argmax(
            [np.sum(rotulo_da_vez[:] == z) / k for z in range(len(classes))]
        )
        rotulo_real = np.argmax(y_teste[i, :])
        if contador_rotulos == rotulo_real:
            contador_acertos += 1
    acuracia = contador_acertos / X_teste.shape[0]
    bp = 1


def dmc(
    X_treino: np.ndarray[Any, np.dtype[Any]],
    y_treino: np.ndarray[Any, np.dtype[Any]],
    X_teste: np.ndarray[Any, np.dtype[Any]],
    y_teste: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
):
    mcs = computar_pontos_medios(X_treino, y_treino, classes)

    for j in range(X_teste.shape[0]):
        X_i = X_teste[j, :]  # iésima amostra de teste


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


def computar_pontos_medios(
    X_treino: np.ndarray[Any, np.dtype[Any]],
    y_treino: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
):
    qtd_classes = len(classes)
    classes_preditoras = computar_vetores_classificacao(classes)
    pontos_medios = []
    for classe_preditora in classes_preditoras:
        pontos_medios.append(
            np.mean(
                X_treino[
                    np.count_nonzero(y_treino[:, :] == classe_preditora, axis=1)
                    == qtd_classes,
                    :,
                ],
                axis=0,
            )
        )
    return pontos_medios


def computar_vetores_classificacao(classes: List[str]):
    vetores_classificacao = []
    qtd_classes = len(classes)
    for classe in classes:
        vetor_preditor = [[-1 for _ in range(qtd_classes)]]
        vetor_preditor[0][classes.index(classe)] = 1
        vetores_classificacao.append(vetor_preditor)
    return vetores_classificacao
