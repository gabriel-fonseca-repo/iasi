from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def carregar_dados(arquivo: str):
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


def processar_dados_especial(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    if X.shape[0] == 3:
        X = X.T
    X_treino = np.concatenate((X[0:1200, :], X[1500:2700, :]))
    y_treino = np.concatenate((y[0:1200, :], y[1500:2700, :]))
    X_teste = np.concatenate((X[1200:1500, :], X[2700:3360, :]))
    y_teste = np.concatenate((y[1200:1500, :], y[2700:3360, :]))

    # (X_treino, y_treino) = embaralhar_dados(X_treino, y_treino)
    # (X_teste, y_teste) = embaralhar_dados(X_teste, y_teste)

    return (X_treino, y_treino, X_teste, y_teste)


def computar_indice_mc(y: int):
    if y == -1:
        return 0
    else:
        return 1


def estatisticas_perceptron(
    ACURACIAS: list, SENSIBILIDADES: list, ESPECIFICIDADE: list
):
    stats = {
        "Estatísticas": ["Acurácia", "Sensibilidade", "Especificidade"],
        "Média": [
            np.mean(ACURACIAS),
            np.mean(SENSIBILIDADES),
            np.mean(ESPECIFICIDADE),
        ],
        "Desvio Padrão": [
            np.std(ACURACIAS),
            np.std(SENSIBILIDADES),
            np.std(ESPECIFICIDADE),
        ],
        "Máximo": [
            np.max(ACURACIAS),
            np.max(SENSIBILIDADES),
            np.max(ESPECIFICIDADE),
        ],
        "Mínimo": [
            np.min(ACURACIAS),
            np.min(SENSIBILIDADES),
            np.min(ESPECIFICIDADE),
        ],
    }

    df = pd.DataFrame(stats)
    df.to_csv("out/rst_perceptron.csv", sep=";")
    plt.figure(figsize=(10, 6))
    plt.bar(df["Estatísticas"], df["Média"], yerr=df["Desvio Padrão"])
    plt.xlabel("Modelo")
    plt.ylabel("Média de EQM")
    plt.title("Estatísticas de performance por Modelo")
    plt.savefig("out/Estatisticas_ols.png")


def plotar_dados(X: np.ndarray[Any, np.dtype[Any]], has_bias: bool = True):
    IDX_0 = 0
    IDX_1 = 1200
    IDX_2 = IDX_1 + 300
    IDX_3 = IDX_2 + 1200
    IDX_4 = IDX_3 + 660

    COL_1 = 1 if has_bias else 0
    COL_2 = 2 if has_bias else 1
    LINES = 3 if has_bias else 2

    if X.shape[0] == LINES:
        X = X.T
    # fmt: off
    plt.scatter(X[IDX_0:IDX_1, COL_1], X[IDX_0:IDX_1, COL_2], color="brown", alpha=0.5, edgecolors="k")
    plt.scatter(X[IDX_1:IDX_2, COL_1], X[IDX_1:IDX_2, COL_2], color="brown", alpha=0.5, edgecolors="k")
    plt.scatter(X[IDX_2:IDX_3, COL_1], X[IDX_2:IDX_3, COL_2], color="orange", alpha=0.5, edgecolors="k")
    plt.scatter(X[IDX_3:IDX_4, COL_1], X[IDX_3:IDX_4, COL_2], color="orange", alpha=0.5, edgecolors="k")
    # fmt: off
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    if X.shape[1] == 2:
        X = X.T
