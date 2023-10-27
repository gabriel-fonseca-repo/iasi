import matplotlib.pyplot as plt
import numpy as np
from util import (
    carregar_dados,
    computar_indice_mc,
    estatisticas_perceptron,
    processar_dados_especial,
    rdc,
    sinal,
)

(X, y) = carregar_dados("data/DataAV2_new.csv")

N, p = X.shape
IDX_0 = 0
IDX_1 = 1200
IDX_2 = IDX_1 + 1200
IDX_3 = IDX_2 + 300
IDX_4 = IDX_3 + 660

# fmt: off
plt.scatter(X[IDX_0:IDX_1, 0], X[IDX_0:IDX_1, 1], color="brown", alpha=0.5, edgecolors="k")
plt.scatter(X[IDX_1:IDX_2, 0], X[IDX_1:IDX_2, 1], color="orange", alpha=0.5, edgecolors="k")
plt.scatter(X[IDX_2:IDX_3, 0], X[IDX_2:IDX_3, 1], color="brown", alpha=0.5, edgecolors="k")
plt.scatter(X[IDX_3:IDX_4, 0], X[IDX_3:IDX_4, 1], color="orange", alpha=0.5, edgecolors="k")
# fmt: off

plt.xlim(-12, 12)
plt.ylim(-12, 12)

X = X.T
X = np.concatenate((-np.ones((1, N)), X), axis=0)
y.shape = (len(y), 1)
W = np.array([[0], [0], [0]])
x1 = np.linspace(-12, 12, N)
x2 = np.zeros((N,))
LR = 0.1

plt.plot(x1, x2, color="red", linewidth=3)

RODADAS = 0
MAX_RODADAS = 100

ACURACIAS = []
SENSIBILIDADES = []
ESPECIFICIDADES = []

while RODADAS < MAX_RODADAS:

    (X_treino, y_treino, X_teste, y_teste) = processar_dados_especial(X, y)

    N, p = X_treino.shape
    N_teste, p_teste = X_teste.shape

    ERRO = True
    EPOCH = 0
    MAX_EPOCH = 100

    while ERRO and EPOCH < MAX_EPOCH:

        ERRO = False
        qtd_erros = 0
        matriz_confusao = np.zeros((2, 2))

        # Fase de treino
        for t in range(N):
            x_t = X_treino[t, :].reshape(3, 1)
            u_t = W.T @ x_t
            y_t = sinal(u_t[0, 0])

            d_t = y_treino[t, 0]
            e_t = int(d_t - y_t)
            W = W + (e_t * x_t * LR) / 2

            if e_t != 0:
                ERRO = True
                qtd_erros += 1
                x2 = -x1 * (W[1, 0] / W[2, 0]) + W[0, 0] / W[2, 0]
        plt.plot(x1, x2, color=rdc(), alpha=0.4)
        plt.pause(0.01)

        # Fase de testes
        for t in range(N_teste):
            x_t = X_teste[t, :].reshape(3, 1)
            u_t = W.T @ x_t
            y_t = sinal(u_t[0, 0])

            d_t = y_teste[t, 0]
            e_t = int(d_t - y_t)

            if e_t != 0:
                ERRO = True
                qtd_erros += 1

            indice_y_mc = computar_indice_mc(int(y_teste[t][0]))
            indice_y_t_mc = computar_indice_mc(y_t)
            matriz_confusao[indice_y_mc, indice_y_t_mc] += 1

        VP: int = matriz_confusao[0, 0]
        VN: int = matriz_confusao[1, 1]
        FP: int = matriz_confusao[0, 1]
        FN: int = matriz_confusao[1, 0]

        ACURACIA = (VP + VN) / (VP + VN + FP + FN)
        SENSIBILIDADE = VP / (VP + FN)
        ESPECIFICIDADE = VN / (VN + FP)

        ACURACIAS.append(ACURACIA)
        SENSIBILIDADES.append(SENSIBILIDADE)
        ESPECIFICIDADES.append(ESPECIFICIDADE)

        print(f"Época {EPOCH + 1} - QTD Erros {qtd_erros} - Amostras {N}")
        EPOCH += 1
    RODADAS += 1

estatisticas_perceptron(ACURACIAS, SENSIBILIDADES, ESPECIFICIDADES)

pass
