import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from util import (
    carregar_dados,
    computar_indice_mc,
    estatisticas,
    plotar_dados,
    printar_progresso,
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


(X, y) = carregar_dados("data/DataAV2_new_2.csv")
plotar_dados(X, has_bias=False)
plt.show()

N, p = X.shape
X = X.T
X = np.concatenate((-np.ones((1, N)), X))

LR = 1e-2
PRECISION = 1e-2

MAX_EPOCH = 300

RODADA = 0
MAX_RODADAS = 100

RODADA_DATA = []

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
    EPOCH_DATA = []

    matriz_confusao_rodada = np.zeros((2, 2), dtype=int)

    plt.clf()
    plotar_dados(X, has_bias=True)

    while EPOCH < MAX_EPOCH and abs(EQM1 - EQM2) > PRECISION:
        matriz_confusao = np.zeros((2, 2), dtype=int)

        printar_progresso(RODADA / MAX_RODADAS)

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
        EQM2 = EQM(X, y, W)

        # plt.plot(x1, x2, color=rdc(), alpha=0.4)
        # plt.pause(0.01)

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

        matriz_confusao_rodada += matriz_confusao

        EPOCH_DATA.append(
            {
                "acuracia": ACURACIA,
                "sensibilidade": SENSIBILIDADE,
                "especificidade": ESPECIFICIDADE,
                "epoch": EPOCH,
                "rodada": RODADA,
                "matriz_confusao": matriz_confusao,
                "x1": x1,
                "x2": x2,
            }
        )

        EPOCH += 1

    RODADA_DATA.append(
        {
            "acuracia": np.mean(ACURACIAS),
            "sensibilidade": np.mean(SENSIBILIDADES),
            "especificidade": np.mean(ESPECIFICIDADES),
            "rodada": RODADA,
            "matriz_confusao": matriz_confusao_rodada,
            "epoch_data": EPOCH_DATA,
        }
    )

    RODADA += 1

# 7. Ao final das rodadas, compute os seguintes resultados para o PS e ADALINE
# A. Acurácia média
# B. Sensibilidade média
# C. Especificidade média
ACURACIAS = [d["acuracia"] for d in RODADA_DATA]
SENSIBILIDADES = [d["sensibilidade"] for d in RODADA_DATA]
ESPECIFICIDADES = [d["especificidade"] for d in RODADA_DATA]
estatisticas(ACURACIAS, SENSIBILIDADES, ESPECIFICIDADES, modelo="Adaline")

MELHOR_RODADA = max(RODADA_DATA, key=lambda x: x["acuracia"])
PIOR_RODADA = min(RODADA_DATA, key=lambda x: x["acuracia"])

plt.show()

# D. Matriz de confusão da melhor rodada
plt.clf()
sns.heatmap(MELHOR_RODADA["matriz_confusao"], annot=True, fmt="d")

# E. Matriz de confusão da pior rodada
plt.clf()
sns.heatmap(PIOR_RODADA["matriz_confusao"], annot=True, fmt="d")

# F. Traçar o hiperplano de separação das duas classes para a melhor e pior rodada
MELHOR_EPOCH = max(MELHOR_RODADA["epoch_data"], key=lambda x: x["acuracia"])
PIOR_EPOCH = min(PIOR_RODADA["epoch_data"], key=lambda x: x["acuracia"])
plt.clf()
plotar_dados(X, has_bias=True)
plt.plot(MELHOR_EPOCH["x1"], MELHOR_EPOCH["x2"], color="black", alpha=0.4)
plt.clf()
plotar_dados(X, has_bias=True)
plt.plot(PIOR_EPOCH["x1"], PIOR_EPOCH["x2"], color="black", alpha=0.4)

pass
