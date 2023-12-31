import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

        q_c = q_neuronios_saida
        self.matriz_confusao_treino = np.zeros((q_c, q_c), dtype=int)
        self.matriz_confusao_teste = np.zeros((q_c, q_c), dtype=int)

        self.acuracias_treino = []
        self.acuracias_teste = []

        self.epocas = []

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

    def get_indices_mc(self, y, i, c):
        rotulo_real = np.argmax(y[:, i].reshape((c, 1)))
        rotulo_predito = np.argmax(self.y[self.q_camadas].reshape((c, 1)))
        return rotulo_real, rotulo_predito

    def taxa_acerto(self, matriz_confusao):
        acertos = 0
        total = 0
        for i in range(len(matriz_confusao)):
            for j in range(len(matriz_confusao)):
                if i == j:
                    acertos += matriz_confusao[i][j]
                total += matriz_confusao[i][j]
        return acertos / total

    def treinar(self, X_treino, y_treino):
        eqm = 1
        epoch = 0
        c, _ = y_treino.shape
        while eqm > self.max_error and epoch < self.max_epoch:
            for i in range(X_treino.shape[1]):
                x_amostra = X_treino[:, i]
                self.forward(x_amostra)

                rotulo_real, rotulo_predito = self.get_indices_mc(y_treino, i, c)
                self.matriz_confusao_treino[rotulo_real][rotulo_predito] += 1

                d = y_treino[:, i]
                self.backward(x_amostra, d)

            eqm = self.calc_eqm(X_treino, y_treino)
            print(f"\rEQM: {eqm:3.3f}% Epoch: {epoch}", end="")
            epoch += 1
        self.acuracias_treino.append(self.taxa_acerto(self.matriz_confusao_treino))
        self.epocas.append(epoch)
        print()

    def treinar_early_stopping(self, X_treino, y_treino):
        eqm = 1
        epoch = 0
        _, N = X_treino.shape
        c, _ = y_treino.shape
        X_validacao = X_treino[:, int(N * 0.8) :]
        y_validacao = y_treino[:, int(N * 0.8) :]
        X_treino = X_treino[:, : int(N * 0.8)]
        y_treino = y_treino[:, : int(N * 0.8)]
        ultimos_5_eqm = []
        while eqm > self.max_error and epoch < self.max_epoch:
            for i in range(X_treino.shape[1]):
                x_amostra = X_treino[:, i]
                self.forward(x_amostra)

                rotulo_real, rotulo_predito = self.get_indices_mc(y_treino, i, c)
                self.matriz_confusao_treino[rotulo_real][rotulo_predito] += 1

                d = y_treino[:, i]
                self.backward(x_amostra, d)

            if epoch % 5 == 0 and epoch > 0:
                eqm_validacao = self.calc_eqm(X_validacao, y_validacao)
                if eqm_validacao > np.mean(ultimos_5_eqm):
                    print(f"\nTendência de crescimento no EQM época: {epoch}.")
                    break
                else:
                    eqm = eqm_validacao
                ultimos_5_eqm = []
            else:
                eqm = self.calc_eqm(X_treino, y_treino)
            ultimos_5_eqm.append(eqm)

            print(f"\rEQM: {eqm:3.3f}% Epoch: {epoch}", end="")
            epoch += 1

        self.acuracias_treino.append(self.taxa_acerto(self.matriz_confusao_treino))
        self.epocas.append(epoch)
        print()

    def testar(self, X_teste, y_teste):
        N, p = X_teste.shape
        c, _ = y_teste.shape
        for i in range(p):
            x_amostra_teste = X_teste[:, i].reshape((N, 1))
            self.forward(x_amostra_teste)
            rotulo_real, rotulo_predito = self.get_indices_mc(y_teste, i, c)
            self.matriz_confusao_teste[rotulo_real][rotulo_predito] += 1
        self.acuracias_teste.append(self.taxa_acerto(self.matriz_confusao_teste))

    def plotar_mc(self, matriz_confusao, fase: str):
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            matriz_confusao,
            annot=True,
            fmt="d",
            cbar=False,
            xticklabels=range(1, 21),
            yticklabels=range(1, 21),
        )
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.savefig(f"out/tc2/MatrizConfusao_MLP_{fase}_{self.q_neuronios[-1]}N.png")
        plt.clf()

    def estatisticas(self, ACURACIAS: list, fase: str, modelo: str = "MLP"):
        ac_as_nparray = np.array(ACURACIAS)
        stats = {
            "Estatísticas": ["Acurácia"],
            "Média": [
                np.mean(ac_as_nparray),
            ],
            "Desvio Padrão": [
                np.std(ac_as_nparray),
            ],
            "Máximo": [
                np.max(ac_as_nparray),
            ],
            "Mínimo": [
                np.min(ac_as_nparray),
            ],
        }

        df = pd.DataFrame(stats)
        df.to_csv(f"out/tc2/rst_{modelo}_{fase}_{self.q_neuronios[-1]}N.csv", sep=";")
        plt.clf()
        plt.boxplot(ACURACIAS)
        plt.title(f"Boxplot das Acurácias - {fase}")
        plt.ylabel("Acurácia")
        plt.savefig(f"out/tc2/BoxPlot_{modelo}_{fase}_{self.q_neuronios[-1]}N.png")
        plt.clf()


q_neuronios_entrada = 30
X, y = get_dados_imagens(q_neuronios_entrada)

# Normalização dos dados
X = 2 * (X / 255) - 1

MATRIZ_CONFUSAO_TREINO_TOTAL = np.zeros((20, 20), dtype=int)
MATRIZ_CONFUSAO_TESTE_TOTAL = np.zeros((20, 20), dtype=int)

ACURACIAS_TREINO = []
ACURACIAS_TESTE = []

QTD_EPOCAS = []

for _ in range(20):
    main_mlp = MLP(
        q_neuronios=[1500, 1300, 1100, 900, 700, 500, 300, 100],
        q_neuronios_saida=20,
        max_epoch=1000,
        max_error=0.003,
        q_neuronios_entrada=q_neuronios_entrada**2,
    )

    # embaralhar os dados
    seed = np.random.permutation(X.shape[1])
    X = X[:, seed]
    y = y[:, seed]

    # dividir em treino e teste 80% e 20%
    X_treino = X[:, : int(X.shape[1] * 0.8)]
    y_treino = y[:, : int(y.shape[1] * 0.8)]
    X_teste = X[:, int(X.shape[1] * 0.8) :]
    y_teste = y[:, int(y.shape[1] * 0.8) :]

    main_mlp.treinar_early_stopping(X_treino, y_treino)
    main_mlp.testar(X_teste, y_teste)

    MATRIZ_CONFUSAO_TREINO_TOTAL += main_mlp.matriz_confusao_treino
    MATRIZ_CONFUSAO_TESTE_TOTAL += main_mlp.matriz_confusao_teste

    ACURACIAS_TREINO.extend(main_mlp.acuracias_treino)
    ACURACIAS_TESTE.extend(main_mlp.acuracias_teste)

    QTD_EPOCAS.extend(main_mlp.epocas)


nome_arquivo_treino = "Treino_Ideal_BP"
nome_arquivo_teste = "Teste_Ideal_BP"

main_mlp.plotar_mc(MATRIZ_CONFUSAO_TREINO_TOTAL, nome_arquivo_treino)
main_mlp.plotar_mc(MATRIZ_CONFUSAO_TESTE_TOTAL, nome_arquivo_teste)

main_mlp.estatisticas(ACURACIAS_TREINO, nome_arquivo_treino)
main_mlp.estatisticas(ACURACIAS_TESTE, nome_arquivo_teste)

mean_epoch = np.mean(QTD_EPOCAS)
print(f"Quantidade média de épocas para atingir convergência: {mean_epoch}")
