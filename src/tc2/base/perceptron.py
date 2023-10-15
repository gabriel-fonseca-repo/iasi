import matplotlib.pyplot as plt
import numpy as np


def sinal(x):
    if x >= 0:
        return 1
    else:
        return -1


X = np.array(
    [[1, 1], [0, 1], [0, 2], [1, 0], [2, 2], [4, 1.5], [1.5, 6], [3, 5], [3, 3], [6, 4]]
)

y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

plt.scatter(X[0:5, 0], X[0:5, 1], color="brown")
plt.scatter(X[5:, 0], X[5:, 1], color="orange")

X = X.T
X = np.concatenate((-np.ones((1, 10)), X), axis=0)
y.shape = (len(y), 1)

W = np.array([[0], [0], [0]])
x1 = np.linspace(-2, 8, 10)

x2 = np.zeros((10,))

plt.plot(x1, x2, color="red", linewidth=3)
plt.xlim(-1, 7)
plt.ylim(-1, 7)

LR = 0.01

Erro = True
while Erro:
    Erro = False
    for t in range(X.shape[1]):
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
