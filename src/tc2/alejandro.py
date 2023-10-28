import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection


class Perceptron(object):
    def __init__(self) -> None:
        self.W = np.array([0.0, 0.0, 0.0])

    def fit(self, LR, X, y, XLine, YLine):
        Erro = True
        X = np.column_stack((-np.ones(X.shape[0]), X))
        epoch = 0
        while Erro and epoch < 300:
            Erro = False
            for t in range(X.shape[0]):
                x_t = X[t]
                u_t = self.W.T @ x_t  # u = W * xK

                y_t = self.sinal(u_t)  # calculando G(u)
                d_t = y[t, 0]  # D sired value for this colum of test
                self.W += LR * (d_t - y_t) * x_t
                if (d_t - y_t) != 0:  # se houver erro:
                    Erro = True

            YLine = -XLine * (self.W[1] / self.W[2]) + self.W[0] / self.W[2]
            plt.plot(XLine, YLine)
            plt.pause(0.05)
            epoch += 1

    def predict(self, X):
        return self.sinal(self.W.T @ X)

    def sinal(self, x):
        if x >= 0:
            return 1
        else:
            return -1


class Adaline(object):
    def __init__(self) -> None:
        self.W = np.array([0.0, 0.0, 0.0])

    def fit(self, LR, X, y, XLine, YLine):
        eT = 0
        last_e = 1
        precision = 1e-3
        epoch = 0
        X = np.column_stack((-np.ones(X.shape[0]), X))
        while abs(eT - last_e) >= precision and epoch < 300:
            last_e = self.EQM(X, y, self.W)
            for t in range(X.shape[0]):
                x_t = X[t]
                u_t = self.W.T @ x_t

                y_t = self.sinal(u_t)
                d_t = y[t, 0]
                self.W += LR * (d_t - y_t) * x_t

            YLine = -XLine * (self.W[1] / self.W[2]) + self.W[0] / self.W[2]
            plt.plot(XLine, YLine)
            plt.pause(0.05)

            epoch += 1
            eT = self.EQM(X, y, self.W)

        print(epoch)

    def predict(self, X):
        return self.sinal(self.W.T @ X)

    def sinal(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def EQM(self, X, Y, W):
        eqm = 0
        for i, x_T in enumerate(X):
            uT = W.T @ x_T
            eqm += (Y[i][0] - uT) ** 2

        return eqm / (2 * X.shape[0])


symbol_marker_map = {
    0: "o",  # Circle marker
    1: "s",  # Square marker
    2: "^",  # Triangle marker
    3: "v",  # Inverted triangle marker
    4: "D",  # Diamond marker
}

color_marker_map = {
    0: "r",  # Circle marker
    1: "g",  # Square marker
    2: "b",  # Triangle marker
    3: "y",  # Inverted triangle marker
    4: "pink",  # Diamond marker
}

class_marker_map = {
    0: "neutro",  # Circle marker
    1: "sorrindo",  # Square marker
    2: "aberto",  # Triangle marker
    3: "surpreso",  # Inverted triangle marker
    4: "Rabugento",  # Diamond marker
}
