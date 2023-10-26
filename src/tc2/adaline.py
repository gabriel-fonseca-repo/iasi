import numpy as np
import matplotlib.pyplot as plt

from util import carregar_dados


def EQM(X, y, w):
    seq = 0
    us = []
    p, N = X.shape
    for t in range(X.shape[1]):
        x_t = X[:, t].reshape(X.shape[0], 1)
        u_t = w.T @ x_t
        us.append(u_t)
        d_t = y[t, 0]
        seq += (d_t - u_t) ** 2

    return seq / (2 * X.shape[1])


(X, y) = carregar_dados("data/DataAV2_O.csv")

N, p = X.shape

X = X.T

X = np.concatenate((-np.ones((1, N)), X))


lr = 1e-2
pr = 0.0000001

maxEpoch = 1000

epoch = 0
EQM1 = 1
EQM2 = 0

while epoch < maxEpoch and abs(EQM1 - EQM2) > pr:
    EQM1 = EQM(X, y, w)
    for t in range(N):
        x_t = X[:, t].reshape(3, 1)
        u_t = w.T @ x_t
        d_t = y[t, 0]
        e_t = d_t - u_t
        w = w + lr * e_t * x_t

    epoch += 1
    EQM2 = EQM(X, y, w)


print(f"Durou {epoch} epócas")
