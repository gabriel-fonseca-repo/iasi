import random as rd

from typing import List, Any
import numpy as np

from modelos import computar_vetores_classificacao, concatenar_uns, eqm, mqo_tikhonov


def definir_melhor_lambda(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    X_t: np.ndarray[Any, np.dtype[Any]],
    y_t: np.ndarray[Any, np.dtype[Any]],
    lbds: list,
):
    X_t = concatenar_uns(X_t)
    melhor_lambda = lbds[0]
    media_lambdas = []
    valores_lambdas = {x: list() for x in lbds}
    for _ in range(1000):
        (X_random, y_random) = embaralhar_dados(X, y)
        for x, _ in valores_lambdas.items():
            modelo_da_vez = mqo_tikhonov(X_random, y_random, x)
            y_pred = X_t @ modelo_da_vez
            valores_lambdas[x].append(eqm(y_pred, y_t))
    for _, eqm_lbd in valores_lambdas.items():
        media_lambdas.append(np.mean(eqm_lbd))
    melhor_lambda = np.argmin(media_lambdas)
    return lbds[melhor_lambda]


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


def separar_classes(
    X: np.ndarray[Any, np.dtype[Any]],
    X_t: np.ndarray[Any, np.dtype[Any]],
):
    N, _ = X.shape
    X_treino_s = X[:, : N - 1]
    X_teste_s = X_t[:, : N - 1]
    return (X_treino_s, X_teste_s)


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
    vetores_classificacao = computar_vetores_classificacao(classes)
    for vetor in vetores_classificacao:
        classes_preditoras.append(np.tile(vetor, (1000, 1)))
    y = np.tile(np.concatenate(classes_preditoras), (10, 1))
    return y


def gerar_cor():
    color = rd.randrange(0, 2**24)
    hex_color = hex(color)
    std_color = "#" + hex_color[2:]
    return std_color
