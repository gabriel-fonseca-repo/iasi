import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from util import get_dados_imagens


class MLP:
    def __init__(
        self,
        q_neuronios,
        q_neuronios_saida,
        max_epoch,
        max_error,
        q_neuronios_entrada,
    ):
        self.q_camadas = len(q_neuronios)
        self.q_neuronios = q_neuronios
        self.q_neuronios_saida = q_neuronios_saida
        self.q_neuronios_entrada = q_neuronios_entrada
        self.max_epoch = max_epoch
        self.max_error = max_error
        self.q_neuronios_entrada = q_neuronios_entrada
        self.W = []
        self.i = [None] * (self.q_camadas + 1)
        self.y = [None] * (self.q_camadas + 1)
        self.delta = [None] * (self.q_camadas + 1)
        self.taxa_aprendizado = 0.05

        self.random_w()

    def random_w(self):
        for i in range(self.q_camadas + 1):
            destino = self.q_neuronios_saida
            origem = self.q_neuronios_entrada
            cur_matrix = []
            if i == 0:
                origem = self.q_neuronios_entrada
            else:
                origem = self.q_neuronios[i - 1]
            if i < self.q_camadas:
                destino = self.q_neuronios[i]
            for _ in range(destino):
                cur_matrix.append(np.random.uniform(-0.5, 0.5, origem + 1))
            self.W.append(np.array(cur_matrix))

    def forward(self, x_amostra):
        for j in range(len(self.W)):
            if j == 0:
                x_amostra.shape = (len(x_amostra), 1)
                x_amostra = np.concatenate((-np.ones((1, 1)), x_amostra), axis=0)
                self.i[j] = self.W[j] @ x_amostra
                self.y[j] = self.g(self.i[j])
            else:
                y_bias = self.y[j - 1]
                y_bias.shape = (len(y_bias), 1)
                y_bias = np.concatenate((-np.ones((1, 1)), y_bias), axis=0)

                self.i[j] = self.W[j] @ y_bias
                self.y[j] = self.g(self.i[j])

    def backward(self, x_amostra, d):
        j = len(self.W) - 1
        d.shape = (len(d), 1)
        x_amostra.shape = (len(x_amostra), 1)
        x_amostra = np.concatenate((-np.ones((1, 1)), x_amostra), axis=0)
        W_b = [None] * (len(self.W) + 1)
        # fmt: off
        while j >= 0:
            if j + 1 == len(self.W):
                self.delta[j] = self.g_linha(self.i[j]) * (d - self.y[j])
                y_bias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                self.W[j] = self.W[j] + self.taxa_aprendizado * (self.delta[j] @ y_bias.T)
            elif j == 0:
                W_b[j + 1] = np.delete(self.W[j + 1].T, 0, 0)
                self.delta[j] = self.g_linha(self.i[j]) * (W_b[j + 1] @ self.delta[j + 1])
                self.W[j] = self.W[j] + self.taxa_aprendizado * (self.delta[j] @ x_amostra.T)
            else:
                W_b[j + 1] = np.delete(self.W[j + 1].T, 0, 0)
                self.delta[j] = self.g_linha(self.i[j]) * (W_b[j + 1] @ self.delta[j + 1])
                y_bias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                self.W[j] = self.W[j] + self.taxa_aprendizado * (self.delta[j] @ y_bias.T)
            j = j - 1
        # fmt: on

    def calc_eqm(self, X_treino, y_treino):
        eqm = 0
        for i in range(X_treino.shape[1]):
            x_amostra = X_treino[:, i]
            self.forward(x_amostra)
            d = y_treino[:, i]
            eqi = 0
            j = 0
            for y in self.y[-1]:
                eqi = eqi + (d[j] - self.y[-1][j][0]) ** 2
                j += 1
            eqm = eqm + eqi
        eqm = eqm / (2 * X_treino.shape[1])
        return eqm

    def g(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def g_linha(self, x):
        return 0.5 * (1 - self.g(x) ** 2)

    def treinar(self, X_treino, y_treino):
        eqm = 1
        epoch = 0
        while eqm > self.max_error and epoch < self.max_epoch:
            for i in range(X_treino.shape[1]):
                x_amostra = X_treino[:, i]
                self.forward(x_amostra)
                d = y_treino[:, i]
                self.backward(x_amostra, d)
            eqm = self.calc_eqm(X_treino, y_treino)
            print(f"EQM: {eqm} Epoch: {epoch}")
            epoch += 1

    def testar(self, X_teste, y_teste):
        N, p = X_teste.shape
        c, _ = y_teste.shape
        matriz_confusao = np.zeros((20, 20), dtype=int)
        for i in range(p):
            x_amostra_teste = X_teste[:, i].reshape((N, 1))
            self.forward(x_amostra_teste)
            rotulo_real = np.argmax(y_teste[:, i].reshape((c, 1)))
            rotulo_predito = np.argmax(self.y[self.q_camadas].reshape((c, 1)))
            matriz_confusao[rotulo_real][rotulo_predito] += 1
        plt.clf()
        sns.heatmap(
            matriz_confusao,
            annot=True,
            fmt="d",
            cbar=False,
        )
        plt.show()


X, y = get_dados_imagens(30)

# Normalização dos dados
X = 2 * (X / 255) - 1

X_treino = X[:, : int(X.shape[1] * 0.8)]
y_treino = y[:, : int(y.shape[1] * 0.8)]
X_teste = X[:, int(X.shape[1] * 0.8) :]
y_teste = y[:, int(y.shape[1] * 0.8) :]

main_mlp = MLP(
    q_neuronios=[30, 30, 30],
    q_neuronios_saida=20,
    max_epoch=1000,
    max_error=0.01,
    q_neuronios_entrada=30 * 30,
)

main_mlp.treinar(X_treino, y_treino)
main_mlp.testar(X_teste, y_teste)
