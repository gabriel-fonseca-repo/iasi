import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt("EMG.csv", delimiter=",")

# neutro  sorrindo  aberto surpreso  rabugento
colors = ["gray", "yellow", "red", "purple", "blue"]

n = 0
y1 = np.tile(np.array([1, -1, -1, -1, -1]), (1000, 1))
y2 = np.tile(np.array([-1, 1, -1, -1, -1]), (1000, 1))
y3 = np.tile(np.array([-1, -1, 1, -1, -1]), (1000, 1))
y4 = np.tile(np.array([-1, -1, -1, 1, -1]), (1000, 1))
y5 = np.tile(np.array([-1, -1, -1, -1, 1]), (1000, 1))
Y_aux = np.concatenate((y1, y2, y3, y4, y5))
Y = np.empty((0, 5))
for i in range(10):
    Y = np.concatenate(Y, Y_aux)
    for j in range(5):
        plt.scatter(
            Data[n : n + 1000, 0],
            Data[n : n + 1000, 1],
            color=colors[j],
            edgecolors="k",
        )
        n += 1000

plt.grid()

N, p = Data.shape

X = np.concatenate((np.ones((N, 1)), Data), axis=1)

seed = np.random.permutation(N)
X_r = X[seed, :]
Y_r = Y[seed, :]

X_treino = X_r[0 : int(N * 0.8) :, :]
Y_treino = Y_r[0 : int(N * 0.8) :, :]

X_teste = X_r[0 : int(N * 0.8) :, :]
Y_teste = Y_r[0 : int(N * 0.8) :, :]

x_axis = np.linspace(-100, 5100, 2000)

W_hat = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ Y_treino

y1 = -W_hat[0, 0] / W_hat[2, 0] - W_hat[1, 0] / W_hat[2, 0] * x_axis
y2 = -W_hat[0, 1] / W_hat[2, 1] - W_hat[1, 1] / W_hat[2, 1] * x_axis
y3 = -W_hat[0, 2] / W_hat[2, 2] - W_hat[1, 2] / W_hat[2, 2] * x_axis
y4 = -W_hat[0, 3] / W_hat[2, 3] - W_hat[1, 3] / W_hat[2, 3] * x_axis
y5 = -W_hat[0, 4] / W_hat[2, 4] - W_hat[1, 4] / W_hat[2, 4] * x_axis

plt.plot(x_axis, y1, color="orange", linewith=2)
plt.plot(x_axis, y2, color="k", linewith=2)
plt.plot(x_axis, y3, color="brown", linewith=2)
plt.plot(x_axis, y4, color="green", linewith=2)
plt.plot(x_axis, y5, color="violet", linewith=2)

Y_hat1 = X_treino @ W_hat

Y_hat2 = X_teste @ W_hat

Y_hat2 = np.argmax(Y_hat2, axis=1)
Y_teste = np.argmax(Y_teste, axis=1)

plt.xlim(-100, 5000)
plt.ylim(-100, 5000)
plt.show()
