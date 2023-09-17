import numpy as np
import matplotlib.pyplot as plt

import os

os.system("clear")

Data = np.loadtxt("data/EMG.csv", delimiter=",")
N, p = Data.shape

classes = ["Neutro", "Sorrindo", "Aberto", "Surpreso", "Rabugento"]
with open("data/EMG.csv", "w") as arquivo:
    for i in range(10):
        for classe in classes:
            for linha in range(1000):
                arquivo.write("," + classe + "\n")


neutro = np.tile(np.array([[1, -1, -1, -1, -1]]), (1000, 1))
sorrindo = np.tile(np.array([[-1, 1, -1, -1, -1]]), (1000, 1))
aberto = np.tile(np.array([[-1, -1, 1, -1, -1]]), (1000, 1))
surpreso = np.tile(np.array([[-1, -1, -1, 1, -1]]), (1000, 1))
rabugento = np.tile(np.array([[-1, -1, -1, -1, 1]]), (1000, 1))

Y = np.tile(np.concatenate((neutro, sorrindo, aberto, surpreso, rabugento)), (10, 1))

s = np.random.permutation(N)

X = Data[s, :]
Y = Y[s, :]

X = np.concatenate((np.ones((N, 1)), X), axis=1)

X_treino = X[0 : int(N * 0.8), :]
Y_treino = Y[0 : int(N * 0.8), :]

X_teste = X[int(N * 0.8) :, :]
Y_teste = Y[int(N * 0.8) :, :]

lb = 0.1
W_hat = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ Y_treino
W_hat_r = np.linalg.pinv(X_treino.T @ X_treino + lb * np.eye(3)) @ X_treino.T @ Y_treino

Y_hat = X_teste @ W_hat

discriminante = np.argmax(Y_hat, axis=1)
discriminante2 = np.argmax(Y_teste, axis=1)

acertos = discriminante == discriminante2
print(np.count_nonzero(acertos) / 10000)


plt.xlim(-100, 4100)
plt.ylim(-100, 4100)
plt.show()
bp = 1
