import sys
import numpy as np
from typing import Any, List
from datetime import datetime

# convert time string to datetime
inicio = datetime.now()


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
    predicoes = []
    N = X_teste.shape[0]
    for i in range(N):
        X_i = X_teste[i, :]
        dists = np.linalg.norm(X_treino - X_i, axis=1)
        k_menores_indices = np.argsort(dists)[0:k]
        k_menores_vizinhos = y_treino[k_menores_indices, :]
        rotulo_da_vez = np.argmax(k_menores_vizinhos, axis=1)
        prob_rotulo = [np.sum(rotulo_da_vez[:] == z) / k for z in range(len(classes))]
        contador_rotulos = np.argmax(prob_rotulo)
        rotulo_real = np.argmax(y_teste[i, :])
        predicoes.append((contador_rotulos, rotulo_real))
    return predicoes


def ta_knn(predicoes: List[tuple], N: int):
    contador_acertos = 0
    for contador_rotulos, rotulo_real in predicoes:
        if contador_rotulos == rotulo_real:
            contador_acertos += 1
    acuracia = contador_acertos / N
    return acuracia


def dmc(
    X_treino: np.ndarray[Any, np.dtype[Any]],
    y_treino: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
):
    N = X_treino.shape[0]
    excludente_classes = X_treino.shape[1] - 1
    X_treino_preditoras = X_treino[:, 0:excludente_classes]
    id_classes = {classe: id for id, classe in enumerate(classes)}
    mcs = computar_pontos_medios(X_treino_preditoras, y_treino, classes)
    rst_treino = np.array([]).reshape((0, 3))
    for j in range(N):
        X_i = X_treino_preditoras[j, :]
        distancias = {
            mcs_k: distancia_do_centroide(X_i, mcs_v) for mcs_k, mcs_v in mcs.items()
        }
        classe_predita = min(distancias, key=distancias.get)
        arr_rst = np.array(
            [[int(X_i[0]), int(X_i[1]), int(id_classes[classe_predita])]]
        )
        rst_treino = np.concatenate((rst_treino, arr_rst))
    return rst_treino


def printar_progresso(valor):
    agora = datetime.now()
    delta = agora - inicio
    print(
        f"\rProgresso de classificação: {valor:.2%}. Tempo decorrido: {int(delta.total_seconds())} segundos.",
        end="",
    )


def ta_dmc(
    X_treino: np.ndarray[Any, np.dtype[Any]], dmc_pred: np.ndarray[Any, np.dtype[Any]]
):
    return np.mean(X_treino == dmc_pred)


def distancia_do_centroide(
    ponto: np.ndarray[Any, np.dtype[Any]],
    centroide: np.ndarray[Any, np.dtype[Any]],
):
    return np.sqrt(np.sum((ponto - centroide) ** 2))


def eqm(y: np.ndarray[Any, np.dtype[Any]], y_teste: np.ndarray[Any, np.dtype[Any]]):
    return np.mean((y_teste - y) ** 2)


def ta_ols(y_teste: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]):
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
    n_classes = len(classes)
    classes_preditoras = computar_vetores_classificacao_com_classes(classes)
    pontos_medios = {}
    for vetor_preditor, classe in classes_preditoras:
        pontos_da_classe = X_treino[
            np.count_nonzero(y_treino[:, :] == vetor_preditor, axis=1) == n_classes, :
        ]
        pontos_medios[classe] = np.mean(pontos_da_classe, axis=0)
    return pontos_medios


def computar_vetores_classificacao(classes: List[str]):
    vetores_classificacao = []
    qtd_classes = len(classes)
    for classe in classes:
        vetor_preditor = [[-1 for _ in range(qtd_classes)]]
        vetor_preditor[0][classes.index(classe)] = 1
        vetores_classificacao.append(vetor_preditor)
    return vetores_classificacao


def computar_vetores_classificacao_com_classes(classes: List[str]):
    vetores_classificacao = []
    qtd_classes = len(classes)
    for classe in classes:
        vetor_preditor = [[-1 for _ in range(qtd_classes)]]
        vetor_preditor[0][classes.index(classe)] = 1
        vetores_classificacao.append((vetor_preditor, classe))
    return vetores_classificacao
