import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron(object):
    def __init__(self) -> None:
        self.W = np.array([0.0, 0.0, 0.0])

    def fit(self, LR, X, y, XLine, YLine):
        Erro = True
        # Terceiro ponto
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

            # YLine = -XLine*(self.W[1] / self.W[2]) + self.W[0] / self.W[2]
            # plt.plot(XLine, YLine)
            # plt.pause(0.05)
            epoch += 1

    def predict(self, X, Y):
        X = np.column_stack((-np.ones(X.shape[0]), X))

        m = np.zeros((2, 2), dtype=int)

        for i, x_t in enumerate(X):
            resp = self.sinal(self.W.T @ x_t)
            resp = 0 if resp <= -1 else 1
            m[resp][Y[i]] += 1
        return m

    def sinal(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def accuracy(self, confusion_matrix):
        vp = confusion_matrix[0][0]
        vn = confusion_matrix[1][1]
        total = np.sum(confusion_matrix)
        return (vp + vn) / total


Labels = ["X", "Y", "D"]
Data = pd.read_csv(
    r"C:\Users\Infinity\Downloads\Telegram Desktop\Paulo Cirilo\AV2\DataAV2.csv",
    header=None,
    names=["x", "y", "Y"],
)


def calculate_perceptron_metrics():
    accuracies = []
    j = 0
    while j < 10:
        X = Data[["x", "y"]].values
        y = Data[["Y"]].values

        seed = np.random.permutation(len(Data))
        X = X[:, seed]
        y = y[:, seed]

        N, p = X.shape

        X_train = X[0 : int(N * 0.8), :]
        y_train = y[0 : int(N * 0.8), :]
        X_test = X[int(N * 0.8) :, :]
        y_test = y[int(N * 0.8) :, :]

        perceptron = Perceptron()
        perceptron.fit(0.001, X_train, y_train, None, None)
        confusion_matrix = perceptron.predict(X_test, y_test)
        accuracy = perceptron.accuracy(confusion_matrix)
        accuracies.append(accuracy)
        j += 1

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    max_accuracy = np.max(accuracies)
    min_accuracy = np.min(accuracies)

    print(f"Acurácia média: {mean_accuracy}")
    print(f"Desvio padrão da acurácia: {std_accuracy}")
    print(f"Maior valor de acurácia: {max_accuracy}")
    print(f"Menor valor de acurácia: {min_accuracy}")


# print(X_train.shape)

LinePrint = np.linspace(-10, 6, 100)
YLinePrint = np.zeros(100)
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.plot(LinePrint, YLinePrint, c="r")

plt.show()

# PERCEPTRON PART

calculate_perceptron_metrics()
