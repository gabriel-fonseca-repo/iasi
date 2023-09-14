from typing import List
import numpy as np
from pyparsing import Any

from modelos import concatenar_uns, eqm, mqo_tikhonov


def definir_melhor_lambda(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    X_t: np.ndarray[Any, np.dtype[Any]],
    y_t: np.ndarray[Any, np.dtype[Any]],
    lbd: list,
):
    X_t = concatenar_uns(X_t)
    melhor_lambda = lbd[0]
    media_lambdas = []
    for x in lbd:
        media_lambdas_x = []
        for i in range(1000):
            modelo_da_vez = mqo_tikhonov(X, y, x)
            y_pred = X_t @ modelo_da_vez
            media_lambdas_x.append(eqm(y_pred, y_t))
        media_lambdas.append(np.mean(media_lambdas_x))
    melhor_lambda = np.argmin(media_lambdas)
    return lbd[melhor_lambda]


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


def carregar_dados_aerogerador(arquivo: str, delimiter=","):
    dados = None
    if arquivo.endswith(".csv"):
        dados = np.genfromtxt(arquivo, delimiter=delimiter)
    else:
        dados = np.genfromtxt(arquivo)
    X = dados[:, 0:1]
    y = dados[:, 1].reshape(X.shape[0], 1)
    return (X, y)


def carregar_dados_aerogerador(arquivo: str):
    dados = ler_arquivo_dados(arquivo)
    X = dados[:, 0:1]
    y = dados[:, 1].reshape(X.shape[0], 1)
    return (X, y)


def carregar_dados_sigmoidais(arquivo: str):
    dados = ler_arquivo_dados(arquivo)
    X = dados[:, 0:2]
    y = dados[:, 2].reshape(X.shape[0], 1)
    return (X, y)


def carregar_dados_emg(arquivo: str):
    dados = ler_arquivo_dados(arquivo)
    return dados


def ler_arquivo_dados(arquivo: str, delimiter=","):
    dados = None
    if arquivo.endswith(".csv") or delimiter != ",":
        dados = np.genfromtxt(arquivo, delimiter=delimiter)
    else:
        dados = np.genfromtxt(arquivo)
    return dados


def calcular_classes_preditoras(classes: List[str]):
    classes_preditoras = []
    qtd_classes = len(classes)
    for classe in classes:
        preditor = [[-1 for i in range(qtd_classes)]]
        preditor[0][classes.index(classe)] = 1
        classes_preditoras.append(np.tile(np.array(preditor), (1000, 1)))
    y = np.tile(np.concatenate(classes_preditoras), (10, 1))
    return y
