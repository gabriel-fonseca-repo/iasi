import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from util import get_dados_imagens

X, y = get_dados_imagens(30)


# class mlp
class MLP:
    def __init__(
        self,
        q_neuronios,
        q_neuronios_saida,
        max_epoch,
        max_error,
        q_neuronios_entrada,
    ):
        # número de camadas ocultas
        self.q_camadas = len(q_neuronios)
        # Array de L posições, cada posição é o número de neurônios na camada
        self.q_neuronios = q_neuronios
        # quantidade de neuronios qNeuroniosEntrada na camada de saida (possui 20 classes)
        self.q_neuronios_saida = q_neuronios_saida
        self.q_neuronios_entrada = q_neuronios_entrada
        # quantidade maxima de epocas
        self.max_epoch = max_epoch
        # definir o erro maximo
        self.max_error = max_error
        # quantidade de neuronios na camada de entrada
        self.q_neuronios_entrada = q_neuronios_entrada
        self.W = []
        self.i = [None] * (self.q_camadas + 1)
        self.y = [None] * (self.q_camadas + 1)
        self.delta = [None] * (self.q_camadas + 1)
        self.taxa_aprendizado = 0.05

        self.random_w()

    def random_w(self):
        #  Inicializar as L + 1 matrizes W com valores aleatórios pequenos (−0.5, 0.5).
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
            W_atual = np.random.random_sample((destino, origem + 1)) - 0.5
            for j in range(destino):
                cur_matrix.append(np.random.uniform(-0.5, 0.5, origem + 1))
            self.W.append(np.array(cur_matrix))

    def forward(self, x_amostra):
        # 1: Receber a amostra xamostra ∈ R^((p+1)×1).
        # 2: j <− 0
        j = 0
        # 3: for cada matriz de peso W em cada uma das L + 1 camadas. do
        for j in range(len(self.W)):
            # 4: if j == 0 then
            if j == 0:
                # 5: i[j] <− W[j] · xamostra
                x_amostra.shape = (len(x_amostra), 1)
                x_amostra = np.concatenate((-np.ones((1, 1)), x_amostra), axis=0)
                self.i[j] = self.W[j] @ x_amostra
                # 6: y[j] <− g(i[j])
                self.y[j] = self.g(self.i[j])

            # 7: else
            else:
                # 8: ybias <− y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = self.y[j - 1]
                y_bias.shape = (len(y_bias), 1)
                y_bias = np.concatenate((-np.ones((1, 1)), y_bias), axis=0)

                # 9: i[j] <− W[j] · ybias
                self.i[j] = self.W[j] @ y_bias
                # 10: y[j] <− g(i[j])
                self.y[j] = self.g(self.i[j])
            # 11: end if
            # 12: j <− j + 1

            # 13: end for

    def backward(self, x_amostra, d):
        # 1: Receber a amostra xamostra e seu rótulo d
        # 2: j <− Quantidade de matrizes W − 1.
        j = len(self.W) - 1
        # 3: while j ≥ 0 do
        d.shape = (len(d), 1)
        x_amostra.shape = (len(x_amostra), 1)
        x_amostra = np.concatenate((-np.ones((1, 1)), x_amostra), axis=0)
        Wb = [None] * (len(self.W) + 1)

        while j >= 0:
            # 4: if j + 1 ==Quantidade de matrizes W, then
            if j + 1 == len(self.W):
                # 5: δ[j] <− g′(i[j]) ◦ (d − y[j]).
                self.delta[j] = self.g_linha(self.i[j]) * (d - self.y[j])
                # 6: ybias <− y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                # 7: W[j] <− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.taxa_aprendizado * (
                    self.delta[j] @ y_bias.T
                )
            # 8: else if j == 0 then
            elif j == 0:
                # 9: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
                Wb[j + 1] = np.delete(self.W[j + 1].T, 0, 0)
                # 10: δ[j] <− g'(i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
                self.delta[j] = self.g_linha(self.i[j]) * (
                    Wb[j + 1] @ self.delta[j + 1]
                )
                # 11: W[j] <− W[j] + η(δ[j] ⊗ xamostra)
                self.W[j] = self.W[j] + self.taxa_aprendizado * (
                    self.delta[j] @ x_amostra.T
                )
            # 12: else
            else:
                # 13: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
                Wb[j + 1] = np.delete(self.W[j + 1].T, 0, 0)
                # 14: δ[j] <− g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
                self.delta[j] = self.g_linha(self.i[j]) * (
                    Wb[j + 1] @ self.delta[j + 1]
                )
                # 15: ybias <− y[j − 1] com adição de −1 na primeira posição do vetor.
                y_bias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                # 16: W[j] <− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.taxa_aprendizado * (
                    self.delta[j] @ y_bias.T
                )
            # 17: end if
            # 18: j <− j − 1
            j = j - 1
        # 19: end while

    def calc_eqm(self, X_treino, y_treino):
        # 1: EQM <− 0
        eqm = 0
        # 2: for Cada amostra em Xtreino do
        for i in range(X_treino.shape[1]):
            # 3: xamostra <− N−ésima amostra de Xtreino.
            x_amostra = X_treino[:, i]
            # 4: Forward(xamostra)
            self.forward(x_amostra)
            # 5: d <− N−ésimo rótulo de Xtreino.
            d = y_treino[:, i]
            # 6: EQI <− 0
            eqi = 0
            j = 0
            # 8: for Cada neurônio na camada de saída do
            for y in self.y[-1]:
                # 9: EQI <− EQI + (d[j] − y[QTD_L − 1][j])2
                eqi = eqi + (d[j] - self.y[-1][j][0]) ** 2
                j += 1
            # 11: end for
            # 12: EQM <− EQM + EQI
            eqm = eqm + eqi
        # 13: end for
        # 14: EQM <− EQM/(2 ∗ QtdAmostrasTreino)
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
            ### Cada Loop desse while é uma epoca
            for i in range(X_treino.shape[1]):
                x_amostra = X_treino[:, i]
                self.forward(x_amostra)
                d = y_treino[:, i]
                self.backward(x_amostra, d)
            eqm = self.calc_eqm(X_treino, y_treino)
            bp = 0
            print(f"EQM: {eqm} Epoch: {epoch}")
            epoch = epoch + 1

    def testar(self, X_teste, y_teste):
        matriz_confusao = np.zeros((20, 20), dtype=int)
        for i in range(X_teste.shape[1]):
            x_amostra = X_teste[:, i]
            self.forward(x_amostra)
            resultado = self.y[-1]
            esperado = y_teste[:, i]
            esperado.shape = (len(esperado), 1)
            classe_predita = np.argmax(resultado)
            classe_real = np.argmax(esperado)
            matriz_confusao[classe_real][classe_predita] += 1
        plt.clf()
        sns.heatmap(
            matriz_confusao,
            annot=True,
            fmt="d",
            cbar=False,
        )
        plt.show()


# normalizar dados:
X = 2 * (X / 255) - 1

seed = np.random.randint(0, 100)
X = np.random.RandomState(seed).permutation(X.T).T
y = np.random.RandomState(seed).permutation(y.T).T

# dividir em treino e teste 80% e 20%
X_treino = X[:, : int(X.shape[1] * 0.8)]
y_treino = y[:, : int(y.shape[1] * 0.8)]
X_teste = X[:, int(X.shape[1] * 0.8) :]
y_teste = y[:, int(y.shape[1] * 0.8) :]

# main_mlp = MLP([40, 25], 20, 1000, 10, 30 * 30)
main_mlp = MLP([100, 50, 25], 20, 1000, 0.001, 30 * 30)

main_mlp.treinar(X_treino, y_treino)

main_mlp.testar(X_treino, y_treino)
main_mlp.testar(X_teste, y_teste)
