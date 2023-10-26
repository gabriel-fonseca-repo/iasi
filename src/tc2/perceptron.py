import matplotlib.pyplot as plt
import numpy as np
from util import (
    carregar_dados,
    computar_indice_mc,
    estatisticas_perceptron,
    processar_dados,
    rdc,
    sinal,
)

(X, y) = carregar_dados("data/DataAV2_O.csv")

N, p = X.shape
m_X = 1500  # int(N * 0.5)
plt.scatter(X[0:m_X, 0], X[0:m_X, 1], color="brown", alpha=0.5, edgecolors="k")
plt.scatter(X[m_X:, 0], X[m_X:, 1], color="orange", alpha=0.1, edgecolors="k")
plt.xlim(-10, 10)
plt.ylim(-10, 10)

X = X.T
X = np.concatenate((-np.ones((1, N)), X), axis=0)
y.shape = (len(y), 1)
W = np.array([[0], [0], [0]])
x1 = np.linspace(-10, 10, N)
x2 = np.zeros((N,))
LR = 0.1

plt.plot(x1, x2, color="red", linewidth=3)

ERRO = True
EPOCH = 0
MAX_EPOCH = 100

ACURACIAS = []
SENSIBILIDADES = []
ESPECIFICIDADES = []

while ERRO and EPOCH < MAX_EPOCH:
    ERRO = False
    w_anterior = W
    qtd_erros = 0
    matriz_confusao = np.zeros((2, 2))

    (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)

    for t in range(N):
        x_t = X[:, t].reshape(3, 1)
        u_t = W.T @ x_t

        y_t = sinal(u_t[0, 0])
        d_t = y[t, 0]
        e_t = int(d_t - y_t)
        W = W + (e_t * x_t * LR) / 2

        indice_y_mc = computar_indice_mc(int(y[t][0]))
        indice_y_t_mc = computar_indice_mc(y_t)
        matriz_confusao[indice_y_mc, indice_y_t_mc] += 1

        if e_t != 0:
            ERRO = True
            qtd_erros += 1
            x2 = -x1 * (W[1, 0] / W[2, 0]) + W[0, 0] / W[2, 0]

    plt.plot(x1, x2, color=rdc(), alpha=0.4)
    plt.pause(0.01)

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

estatisticas_perceptron(ACURACIAS, SENSIBILIDADES, ESPECIFICIDADES)

pass
