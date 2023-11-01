import matplotlib.pyplot as plt
import numpy as np

from base.OrganizaImagens_Parte2 import get_dados_imagens

X, Y = get_dados_imagens(30)


# class mlp
class MLP:
    def __init__(
        self,
        qNeuronios,
        qNeuroniosSaida,
        maxEpoch,
        maxError,
        qNeuroniosEntrada,
    ):
        # número de camadas ocultas
        self.qCamadas = len(qNeuronios)
        # Array de L posições, cada posição é o número de neurônios na camada
        self.qNeuronios = qNeuronios
        # quantidade de neuronios qNeuroniosEntrada na camada de saida (possui 20 classes)
        self.qNeuroniosSaida = qNeuroniosSaida
        self.qNeuroniosEntrada = qNeuroniosEntrada
        # quantidade maxima de epocas
        self.maxEpoch = maxEpoch
        # definir o erro maximo
        self.maxError = maxError
        # quantidade de neuronios na camada de entrada
        self.qNeuroniosEntrada = qNeuroniosEntrada
        self.W = []
        self.i = [None] * (self.qCamadas + 1)
        self.y = [None] * (self.qCamadas + 1)
        self.delta = [None] * (self.qCamadas + 1)
        self.taxaAprendizado = 0.05

        self.wAleatorio()

    def wAleatorio(self):
        #  Inicializar as L + 1 matrizes W com valores aleatórios pequenos (−0.5, 0.5).
        for i in range(self.qCamadas + 1):
            destino = self.qNeuroniosSaida
            origem = self.qNeuroniosEntrada

            curMatrix = []

            if i == 0:
                origem = self.qNeuroniosEntrada
            else:
                origem = self.qNeuronios[i - 1]

            if i < self.qCamadas:
                destino = self.qNeuronios[i]

            W_atual = np.random.random_sample((destino, origem + 1)) - 0.5

            for j in range(destino):
                curMatrix.append(np.random.uniform(-0.5, 0.5, origem + 1))

            self.W.append(np.array(curMatrix))

    def forward(self, xamostra):
        # 1: Receber a amostra xamostra ∈ R^((p+1)×1).
        # 2: j <− 0
        j = 0
        # 3: for cada matriz de peso W em cada uma das L + 1 camadas. do
        for j in range(len(self.W)):
            # 4: if j == 0 then
            if j == 0:
                # 5: i[j] <− W[j] · xamostra
                xamostra.shape = (len(xamostra), 1)
                xamostra = np.concatenate((-np.ones((1, 1)), xamostra), axis=0)
                self.i[j] = self.W[j] @ xamostra
                # 6: y[j] <− g(i[j])
                self.y[j] = self.g(self.i[j])

            # 7: else
            else:
                # 8: ybias <− y[j − 1] com adição de −1 na primeira posição do vetor.
                ybias = self.y[j - 1]
                ybias.shape = (len(ybias), 1)
                ybias = np.concatenate((-np.ones((1, 1)), ybias), axis=0)

                # 9: i[j] <− W[j] · ybias
                self.i[j] = self.W[j] @ ybias
                # 10: y[j] <− g(i[j])
                self.y[j] = self.g(self.i[j])
            # 11: end if
            # 12: j <− j + 1

            # 13: end for

    def backward(self, xamostra, d):
        # 1: Receber a amostra xamostra e seu rótulo d
        # 2: j <− Quantidade de matrizes W − 1.
        j = len(self.W) - 1
        # 3: while j ≥ 0 do
        d.shape = (len(d), 1)
        xamostra.shape = (len(xamostra), 1)
        xamostra = np.concatenate((-np.ones((1, 1)), xamostra), axis=0)
        Wb = [None] * (len(self.W) + 1)

        while j >= 0:
            # 4: if j + 1 ==Quantidade de matrizes W, then
            if j + 1 == len(self.W):
                # 5: δ[j] <− g′(i[j]) ◦ (d − y[j]).
                self.delta[j] = self.g_linha(self.i[j]) * (d - self.y[j])
                # 6: ybias <− y[j − 1] com adição de −1 na primeira posição do vetor.
                ybias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                # 7: W[j] <− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.taxaAprendizado * (self.delta[j] @ ybias.T)
            # 8: else if j == 0 then
            elif j == 0:
                # 9: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
                Wb[j + 1] = np.delete(self.W[j + 1].T, 0, 0)
                # 10: δ[j] <− g'(i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
                self.delta[j] = self.g_linha(self.i[j]) * (
                    Wb[j + 1] @ self.delta[j + 1]
                )
                # 11: W[j] <− W[j] + η(δ[j] ⊗ xamostra)
                self.W[j] = self.W[j] + self.taxaAprendizado * (
                    self.delta[j] @ xamostra.T
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
                ybias = np.concatenate((-np.ones((1, 1)), self.y[j - 1]), axis=0)
                # 16: W[j] <− W[j] + η(δ[j] ⊗ ybias)
                self.W[j] = self.W[j] + self.taxaAprendizado * (self.delta[j] @ ybias.T)
            # 17: end if
            # 18: j <− j − 1
            j = j - 1
        # 19: end while

    def calcularEQM(self, Xtreino, Ytreino):
        # 1: EQM <− 0
        eqm = 0
        # 2: for Cada amostra em Xtreino do
        for i in range(Xtreino.shape[1]):
            # 3: xamostra <− N−ésima amostra de Xtreino.
            xamostra = Xtreino[:, i]
            # 4: Forward(xamostra)
            self.forward(xamostra)
            # 5: d <− N−ésimo rótulo de Xtreino.
            d = Ytreino[:, i]
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
        eqm = eqm / (2 * Xtreino.shape[1])
        return eqm

    def g(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def g_linha(self, x):
        return 0.5 * (1 - self.g(x) ** 2)

    def treinar(self, Xtreino, Ytreino):
        eqm = 1
        epoch = 0
        while eqm > self.maxError and epoch < self.maxEpoch:
            ### Cada Loop desse while é uma epoca
            for i in range(Xtreino.shape[1]):
                xamostra = Xtreino[:, i]
                self.forward(xamostra)
                d = Ytreino[:, i]
                self.backward(xamostra, d)
            eqm = self.calcularEQM(Xtreino, Ytreino)
            bp = 0
            print(f"EQM: {eqm} Epoch: {epoch}")
            epoch = epoch + 1


# normalizar dados:


# dividir em treino e teste 80% e 20%

X = 2 * (X / 255) - 1

Xtreino = X[:, : int(X.shape[1] * 0.8)]
Ytreino = Y[:, : int(Y.shape[1] * 0.8)]
Xteste = X[:, int(X.shape[1] * 0.8) :]
Yteste = Y[:, int(Y.shape[1] * 0.8) :]

mainMlp = MLP([30, 30, 30], 20, 1000, 0.01, 30 * 30)

mainMlp.treinar(Xtreino, Ytreino)
