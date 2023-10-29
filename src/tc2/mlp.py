import matplotlib.pyplot as plt
import numpy as np

from util import get_dados_imagens


# class mlp
class MLP(object):
    def __init__(
        self,
        q_camadas,
        q_neuronios,
        q_neuronios_saida,
        max_epoch,
        max_error,
        q_neuronios_entrada,
    ):
        # número de camadas ocultas
        self.q_camadas = q_camadas
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
        self.i = np.empty((q_camadas + 1, 0))
        self.y = np.empty((q_camadas + 1, 0))
        self.delta = np.empty((q_camadas + 1, 0))

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

            for j in range(destino):
                cur_matrix.append(np.random.uniform(-0.5, 0.5, origem))

            self.W.append(np.array(cur_matrix))

    def forward(self, xamostra):
        # 1: Receber a amostra xamostra ∈ R^((p+1)×1).
        # 2: j ←− 0
        j = 0
        # 3: for cada matriz de peso W em cada uma das L + 1 camadas. do
        for W in self.W:
            # 4: if j == 0 then
            if j == 0:
                # 5: i[j] ←− W[j] · xamostra
                self.i[j] = W[j] @ xamostra
                # 6: y[j] ←− g(i[j])
                self.y[j] = self.g(self.i[j])
            # 7: else
            else:
                # 8: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
                ybias = np.insert(self.y[j - 1], 0, -1)
                # 9: i[j] ←− W[j] · ybias
                self.i[j] = W[j] @ ybias
                # 10: y[j] ←− g(i[j])
                self.y[j] = self.g(self.i[j])
            # 11: end if
            # 12: j ←− j + 1
            j = j + 1
            # 13: end for

    def backward(self, xamostra, d):
        # 1: Receber a amostra xamostra e seu rótulo d
        # 2: j ←− Quantidade de matrizes W − 1.
        j = len(self.W) - 1
        # 3: while j ≥ 0 do
        while j >= 0:
            # 4: if j + 1 ==Quantidade de matrizes W, then
            if j + 1 == len(self.W):
                # 5: δ[j] ←− g′(i[j]) ◦ (d − y[j]).
                self.delta[j] = self.g_linha(self.i[j]) * (d - self.y[j])
                # 6: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
                ybias = np.append(
                    -np.ones((1, self.y[j - 1].shape[1])), self.y[j - 1], axis=0
                )
                # 7: W[j] ←− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.eta * (self.delta[j] @ ybias.T)
            # 8: else if j == 0 then
            elif j == 0:
                # 9: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
                Wb = np.delete(self.W[j + 1].T, 0, 0)
                # 10: δ[j] ←− g'(i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
                self.delta[j] = self.g_linha(self.i[j]) * (Wb @ self.delta[j + 1])
                # 11: W[j] ←− W[j] + η(δ[j] ⊗ xamostra)
                self.W[j] = self.W[j] + self.eta * (self.delta[j] @ xamostra.T)
            # 12: else
            else:
                # 13: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
                Wb = np.delete(self.W[j + 1].T, 0, 0)
                # 14: δ[j] ←− g′(i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
                self.delta[j] = self.g_linha(self.i[j]) * (Wb @ self.delta[j + 1])
                # 15: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
                ybias = np.append(
                    -np.ones((1, self.y[j - 1].shape[1])), self.y[j - 1], axis=0
                )
                # 16: W[j] ←− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.eta * (self.delta[j] @ ybias.T)
            # 17: end if
            # 18: j ←− j − 1
            j = j - 1
        # 19: end while

    def calcular_eqm(self, X_treino, y_treino):
        # 1: EQM ←− 0
        eqm = 0
        # 2: for Cada amostra em Xtreino do
        for i in range(X_treino.shape[1]):
            # 3: xamostra ←− N−ésima amostra de Xtreino.
            x_amostra = X_treino[:, i]
            # 4: Forward(xamostra)
            self.forward(x_amostra)
            # 5: d ←− N−ésimo rótulo de Xtreino.
            d = y_treino[:, i]
            # 6: EQI ←− 0
            eqi = 0
            # 7: j ←− 0
            j = 0
            # 8: for Cada neurônio na camada de saída do
            for y in self.y[-1]:
                # 9: EQI ←− EQI + (d[j] − y[QTD_L − 1][j])2
                eqi = eqi + (d[j] - y) ** 2
                # 10: j ←− j + 1
                j = j + 1
            # 11: end for
            # 12: EQM ←− EQM + EQI
            eqm = eqm + eqi
        # 13: end for
        # 14: EQM ←− EQM/(2 ∗ QtdAmostrasTreino)
        eqm = eqm / (2 * X_treino.shape[1])

    def g(self, x):
        return 1 / (1 + np.exp(-x))

    def g_linha(self, x):
        return self.g(x)

    def treinar(self, X_treino, y_treino):
        eqm = 1
        epoch = 0
        while eqm > self.max_error and epoch < self.max_epoch:
            ### Cada Loop desse while é uma epoca
            for i in range(X_treino.shape[1]):
                xamostra = X_treino[:, i]
                self.forward(xamostra)
                d = y_treino[:, i]
                self.backward(xamostra, d)
            eqm = self.calcular_eqm()
            print(f"EQM: {eqm} Epoch: {epoch}")
            epoch = epoch + 1


X, Y = get_dados_imagens(60)

# dividir em treino e teste 80% e 20%
X_treino = X[:, : int(X.shape[1] * 0.8)]
y_treino = Y[:, : int(Y.shape[1] * 0.8)]
X_teste = X[:, int(X.shape[1] * 0.8) :]
y_teste = Y[:, int(Y.shape[1] * 0.8) :]

mainMlp = MLP(5, [30, 30, 30, 30, 30], 20, 1000, 0.01, 3600)

mainMlp.treinar(X_treino, y_treino)
