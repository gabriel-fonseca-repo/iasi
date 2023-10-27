import numpy as np

from util import carregar_dados


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

N, p = X.shape

X = X.T

X = np.concatenate((-np.ones((1, N)), X))


LR = 1e-2
PR = 0.0000001

MAX_EPOCH = 1000

EPOCH = 0
EQM1 = 1
EQM2 = 0

while EPOCH < MAX_EPOCH and abs(EQM1 - EQM2) > PR:
    EQM1 = EQM(X, y, W)

    for t in range(N):
        x_t = X[:, t].reshape(3, 1)
        u_t = W.T @ x_t
        d_t = y[t, 0]
        e_t = d_t - u_t
        W = W + LR * e_t * x_t

    EQM2 = EQM(X, y, W)

    EPOCH += 1


print(f"Durou {EPOCH} epócas")
