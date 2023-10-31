import cv2
import seaborn as sns
import datetime as dt
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inicio = dt.datetime.now()


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
    if X.shape[0] == 3:
        X = X.T
    N, _ = X.shape

    (X_random, y_random) = embaralhar_dados(X, y)
    X_treino = X_random[0 : int(N * 0.8), :]
    y_treino = y_random[0 : int(N * 0.8), :]
    X_teste = X_random[int(N * 0.8) :, :]
    y_teste = y_random[int(N * 0.8) :, :]
    return (X_treino, y_treino, X_teste, y_teste)


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


def estatisticas(
    ACURACIAS: list, SENSIBILIDADES: list, ESPECIFICIDADE: list, modelo: str = ""
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
    df.to_csv(f"out/tc2/rst_{modelo}.csv", sep=";")
    # plt.figure(figsize=(10, 6))
    # plt.bar(df["Estatísticas"], df["Média"], yerr=df["Desvio Padrão"])
    # plt.xlabel("Modelo")
    # plt.ylabel("Estatística")
    # plt.title("Estatísticas de performance por Modelo")
    # plt.savefig(f"out/Estatisticas_{modelo}.png")


def exportar_graficos(
    X: np.ndarray[Any, np.dtype[Any]], RODADA_DATA: List[Dict], modelo: str
):
    # 7. Ao final das rodadas, compute os seguintes resultados para o PS e ADALINE
    # A. Acurácia média
    # B. Sensibilidade média
    # C. Especificidade média
    ACURACIAS = [d["acuracia"] for d in RODADA_DATA]
    SENSIBILIDADES = [d["sensibilidade"] for d in RODADA_DATA]
    ESPECIFICIDADES = [d["especificidade"] for d in RODADA_DATA]
    estatisticas(ACURACIAS, SENSIBILIDADES, ESPECIFICIDADES, modelo=modelo)

    MELHOR_RODADA = max(RODADA_DATA, key=lambda x: x["acuracia"])
    PIOR_RODADA = min(RODADA_DATA, key=lambda x: x["acuracia"])

    # D. Matriz de confusão da melhor rodada
    plt.clf()
    sns.heatmap(
        MELHOR_RODADA["matriz_confusao"],
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["-1", "1"],
        yticklabels=["-1", "1"],
    )
    plt.savefig(f"out/tc2/MatrizConfusao_{modelo}_MelhorRodada.png")

    # E. Matriz de confusão da pior rodada
    plt.clf()
    sns.heatmap(
        PIOR_RODADA["matriz_confusao"],
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["-1", "1"],
        yticklabels=["-1", "1"],
    )
    plt.savefig(f"out/tc2/MatrizConfusao_{modelo}_PiorRodada.png")

    # F. Traçar o hiperplano de separação das duas classes para a melhor e pior rodada
    plt.clf()
    plotar_dados(X, has_bias=True)
    plt.plot(MELHOR_RODADA["x1"], MELHOR_RODADA["x2"], color="black", alpha=0.4)
    plt.savefig(f"out/tc2/Hiperplano_{modelo}_MelhorRodada.png")
    plt.clf()
    plotar_dados(X, has_bias=True)
    plt.plot(PIOR_RODADA["x1"], PIOR_RODADA["x2"], color="black", alpha=0.4)
    plt.savefig(f"out/tc2/Hiperplano_{modelo}_PiorRodada.png")


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


def printar_progresso(valor):
    agora = dt.datetime.now()
    delta = agora - inicio
    print(
        f"\rProgresso de classificação: {valor:.2%}. Tempo decorrido: {int(delta.total_seconds())} segundos.",
        end="",
    )


def get_dados_imagens(red):
    folderRoot = "data/tc2/faces/"  # MODIFIQUE para o caminho do conjunto de dados no seu computador.
    individual = [
        "an2i",
        "at33",
        "boland",
        "bpm",
        "ch4f",
        "cheyer",
        "choon",
        "danieln",
        "glickman",
        "karyadi",
        "kawamura",
        "kk49",
        "megak",
        "mitchell",
        "night",
        "phoebe",
        "saavik",
        "steffi",
        "sz24",
        "tammo",
    ]  # os 20 sujeitos no conjunto de dados.
    expressoes = [
        "_left_angry_open",
        "_left_angry_sunglasses",
        "_left_happy_open",
        "_left_happy_sunglasses",
        "_left_neutral_open",
        "_left_neutral_sunglasses",
        "_left_sad_open",
        "_left_sad_sunglasses",
        "_right_angry_open",
        "_right_angry_sunglasses",
        "_right_happy_open",
        "_right_happy_sunglasses",
        "_right_neutral_open",
        "_right_neutral_sunglasses",
        "_right_sad_open",
        "_right_sad_sunglasses",
        "_straight_angry_open",
        "_straight_angry_sunglasses",
        "_straight_happy_open",
        "_straight_happy_sunglasses",
        "_straight_neutral_open",
        "_straight_neutral_sunglasses",
        "_straight_sad_open",
        "_straight_sad_sunglasses",
        "_up_angry_open",
        "_up_angry_sunglasses",
        "_up_happy_open",
        "_up_happy_sunglasses",
        "_up_neutral_open",
        "_up_neutral_sunglasses",
        "_up_sad_open",
        "_up_sad_sunglasses",
    ]
    QtdIndividuos = len(individual)
    QtdExpressoes = len(expressoes)
    X = np.empty((red * red, 0))
    Y = np.empty((QtdIndividuos, 0))

    for i in range(QtdIndividuos):
        for j in range(QtdExpressoes):
            path = folderRoot + individual[i] + "/" + individual[i] + expressoes[j]
            PgmImg = cv2.imread(path + ".pgm", cv2.IMREAD_GRAYSCALE)
            if PgmImg is None:
                PgmImg = cv2.imread(path + ".bad", cv2.IMREAD_GRAYSCALE)

            ResizedImg = cv2.resize(PgmImg, (red, red))

            VectorNormalized = ResizedImg.flatten("F")
            ROT = -np.ones((QtdIndividuos, 1))
            ROT[i, 0] = 1

            # cv2.imshow("Foto", PgmImg)
            # cv2.waitKey(0)

            VectorNormalized.shape = (len(VectorNormalized), 1)
            X = np.append(X, VectorNormalized, axis=1)
            Y = np.append(Y, ROT, axis=1)
    return X, Y
