import matplotlib.pyplot as plt
import numpy as np


def carregar_dados_sigmoidais(arquivo: str):
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


(X, y) = carregar_dados_sigmoidais("data/DataAV2.csv")

N, p = X.shape
m_X = int(N * 0.5)
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

TA_POR_EPOCH = []
MAX_EPOCH = 100


while ERRO and EPOCH < MAX_EPOCH:
    ERRO = False
    w_anterior = W
    qtd_erros = 0

    for t in range(N):
        x_t = X[:, t].reshape(3, 1)
        u_t = W.T @ x_t

        y_t = sinal(u_t[0, 0])
        d_t = y[t, 0]
        e_t = int(d_t - y_t)
        W = W + (e_t * x_t * LR) / 2
        # W = W + LR * e_t * x_t
        if e_t != 0:
            ERRO = True
            qtd_erros += 1
            x2 = -x1 * (W[1, 0] / W[2, 0]) + W[0, 0] / W[2, 0]

    plt.plot(x1, x2, color=rdc(), alpha=0.4)
    plt.pause(0.01)
    TA_POR_EPOCH.append(qtd_erros)
    print(f"Época ({EPOCH}) - QTD Erros {qtd_erros} - Amostras {N}")
    EPOCH += 1

pass
