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


(X, y) = carregar_dados_sigmoidais("data/DataAV2.csv")

N, p = X.shape
metade_X = int(N * 0.5)
plt.scatter(X[0:metade_X, 0], X[0:metade_X, 1], color="brown", alpha=0.5)
plt.scatter(X[metade_X:, 0], X[metade_X:, 1], color="orange", alpha=0.1)

X = X.T
X = np.concatenate((-np.ones((1, N)), X), axis=0)
y.shape = (len(y), 1)
W = np.array([[0], [0], [0]])
x1 = np.linspace(-2, 8, N)
x2 = np.zeros((N,))
LR = 0.01

plt.plot(x1, x2, color="red", linewidth=3)


Erro = True
while Erro:
    Erro = False
    for t in range(N):
        x_t = X[:, t].reshape(3, 1)
        u_t = W.T @ x_t

        y_t = sinal(u_t[0, 0])
        d_t = y[t, 0]
        W = W + LR * (d_t - y_t) * x_t
        if (d_t - y_t) != 0:
            Erro = True
            x2 = -x1 * (W[1, 0] / W[2, 0]) + W[0, 0] / W[2, 0]

            plt.plot(x1, x2, color="k", alpha=0.4)
            plt.pause(0.2)
