import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt(".csv", delimiter=",")

plt.scatter(Data[:, 0], Data[:, 1], edgecolors="k")

# X.shape[0] -> número de amostras
# X.xhape[1] -> preditores

x1 = Data[:, 0].reshape(Data.shape[0], 1)
y = Data[:, 1].reshape(Data.shape[0], 1)

# adicionar o interceptor
uns = np.ones(x1.shape)
X = np.concatenate((uns, x1), axis=1)

# geração de dados aleatórios entre 0 e o número de amostras
seed = np.random.permutation(X.shape[0])

# embaralhamento dos dados
Xr = X[seed, :]
yr = y[seed, :]

# foram pegues 80% dos dados para treino
Xtreino = Xr[0 : int(X.shape[0] * 0.8), :]
ytreino = yr[0 : int(X.shape[0] * 0.8), :]

plt.scatter(Xtreino[:, 1], ytreino[:, 0], edgecolors="k", color="green", alpha=0.3)

# Modelo estimativo
b_hat = np.linalg.pinv(Xtreino.T @ Xtreino) @ Xtreino.T @ ytreino

# foram pegues 20% dos dados para treino
Xteste = Xr[int(X.shape[0] * 0.8) :, :]
yteste = yr[int(X.shape[0] * 0.8) :, :]

plt.scatter(Xteste[:, 1], yteste[:, 1], edgecolors="k", color="pink")

mu = np.mean(ytreino)
b_media = np.array([mu], [0])

x_axis = np.linspace(0, 200, 1000).reshape(1000, 1)

uns = np.ones((1000, 1))

X_axis = np.concatenate((uns, x_axis), axis=1)

y_hatg = X_axis @ b_hat
y_hat_mu = X_axis @ b_media

# Modelo de teste da estimação
y_hat = Xteste @ b_hat
y_hat_m = Xteste @ b_media

plt.plot(
    x_axis,
    y_hatg,
)

plt.scatter(Xteste[:, 1], y_hat_m[:, 0], edgecolors="k", color="gray", s=90)

# plt.show()

residuos1 = np.sum((y_hat - yteste) ** 2)
residuos2 = np.sum((y_hat_m - yteste) ** 2)
