import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from util import (
    carregar_dados,
    computar_indice_mc,
    estatisticas,
    plotar_dados,
    processar_dados,
    rdc,
    sinal,
)

os.system("clear")


def EQM(X, y, W):
    seq = 0
    us = []
    p, N = X.shape
    for t in range(X.shape[1]):
        x_t = X[:, t].reshape(X.shape[0], 1)
        u_t = W.T @ x_t
        us.append(u_t)
        d_t = y[t, 0]
        seq += (d_t - u_t) ** 2
    return seq / (2 * X.shape[1])


(X, y) = carregar_dados("data/DataAV2_O.csv")
plotar_dados(X, has_bias=False)

N, p = X.shape
X = X.T
X = np.concatenate((-np.ones((1, N)), X))

LR = 1e-2
PRECISION = 1e-2

MAX_EPOCH = 1000

RODADA = 0
MAX_RODADAS = 100

RODADAS_DATA = []

while RODADA < MAX_RODADAS:
    (X_treino, y_treino, X_teste, y_teste) = processar_dados(X, y)

    N, p = X_treino.shape
    N_teste, p_teste = X_teste.shape
    X_treino, X_teste = X_treino.T, X_teste.T

    W = np.random.rand(p, 1)
    x1 = np.linspace(-12, 12, N)
    x2 = np.zeros((N,))
    LR = 0.1

    ERRO = True
    QTD_ERROS = 0
    EPOCH = 0

    EQM1 = 1
    EQM2 = 0

    ACURACIAS = []
    SENSIBILIDADES = []
    ESPECIFICIDADES = []

    plt.cla()
    plotar_dados(X, has_bias=True)

    while EPOCH < MAX_EPOCH and abs(EQM1 - EQM2) > PRECISION:
        matriz_confusao = np.zeros((2, 2))

        # Fase de treinamento
        EQM1 = EQM(X, y, W)
        for t in range(N):
            x_t = X_treino[:, t].reshape(3, 1)
            u_t = W.T @ x_t
            d_t = y_treino[t, 0]
            y_t = sinal(u_t[0, 0])
            e_t = d_t - y_t
            W = W + (LR * e_t * x_t)

            if y_t != d_t:
                x2 = -x1 * (W[1, 0] / W[2, 0]) + W[0, 0] / W[2, 0]

        plt.plot(x1, x2, color=rdc(), alpha=0.4)
        plt.pause(0.01)

        EQM2 = EQM(X, y, W)

        # Fase de testes
        for t in range(N_teste):
            x_t = X_teste[:, t].reshape(3, 1)
            u_t = W.T @ x_t
            y_t = sinal(u_t[0, 0])

            y_real = computar_indice_mc(int(y_teste[t][0]))
            y_predito = computar_indice_mc(y_t)
            matriz_confusao[y_predito, y_real] += 1

        VN: int = matriz_confusao[0, 0]
        VP: int = matriz_confusao[1, 1]
        FN: int = matriz_confusao[0, 1]
        FP: int = matriz_confusao[1, 0]

        ACURACIA = (VP + VN) / (VP + VN + FP + FN)
        SENSIBILIDADE = VP / (VP + FN)
        ESPECIFICIDADE = VN / (VN + FP)

        ACURACIAS.append(ACURACIA)
        SENSIBILIDADES.append(SENSIBILIDADE)
        ESPECIFICIDADES.append(ESPECIFICIDADE)

        EPOCH += 1

    RODADAS_DATA.append(
        {
            "acuracia": np.mean(ACURACIAS),
            "sensibilidade": np.mean(SENSIBILIDADES),
            "especificidade": np.mean(ESPECIFICIDADES),
            "rodada": RODADA,
            "matriz_confusao": matriz_confusao,
            "x1": x1,
            "x2": x2,
        }
    )

    RODADA += 1

ACURACIAS = [d["acuracia"] for d in RODADAS_DATA]
SENSIBILIDADES = [d["sensibilidade"] for d in RODADAS_DATA]
ESPECIFICIDADES = [d["especificidade"] for d in RODADAS_DATA]

MELHOR_RODADA = max(RODADAS_DATA, key=lambda x: x["acuracia"])
PIOR_RODADA = min(RODADAS_DATA, key=lambda x: x["acuracia"])

plt.cla()
plotar_dados(X, has_bias=True)
plt.plot(MELHOR_RODADA["x1"], MELHOR_RODADA["x2"], color="black", alpha=0.4)
plt.show()

plt.cla()
plotar_dados(X, has_bias=True)
plt.plot(PIOR_RODADA["x1"], PIOR_RODADA["x2"], color="black", alpha=0.4)
plt.show()

sns.heatmap(MELHOR_RODADA["matriz_confusao"], annot=True)
sns.heatmap(PIOR_RODADA["matriz_confusao"], annot=True)

estatisticas(ACURACIAS, SENSIBILIDADES, ESPECIFICIDADES, modelo="Adaline")

pass
