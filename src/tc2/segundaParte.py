import matplotlib.pyplot as plt
import numpy as np
from OrganizaImagens_Parte2 import getDadosImagens

X, Y = getDadosImagens(60)

# MLP
L = 5  # número de camadas ocultas
# array de L posições, cada posição é o número de neurônios na camada
# 1, 2, ..., L
qNeuronios = [10, 10, 10, 10, 10, 10]
# Definir a quantidade de neuronios m na camada de saida (possui 20 classes)
m = 20
# definir a taxa de aprendizado
n = 0.1
# quantidade maxima de epocas
maxEpoch = 1000
# definir o erro maximo
maxError = 0.01
# Criar uma lista (list) dos elementos: W, i, y, δ cada uma com L + 1 posições
# (uma para cada camada)
W = [] # lista de matrizes de pesos. W[l] é a matriz de pesos da camada l. Tendo dimensionamento ql x ql+1
i = []
y = []
delta = []

#  Inicializar as L + 1 matrizes W com valores aleatórios pequenos (−0.5, 0.5).
#  A matriz W[l] possui ql linhas e ql+1 colunas, onde ql é o número de
#  neurônios na camada l
for l in range(L + 1):
    if l == 0:
        W.append(np.random.uniform(-0.5, 0.5, (qNeuronios[l], X.shape[0])))
    else:
        W.append(np.random.uniform(-0.5, 0.5, (qNeuronios[l], qNeuronios[l - 1])))

# Receber os dados de treinamento com a ordem Xtreino e Ytreino
Xtreino = X
Ytreino = Y

# Adicionar o vetor linha de −1 na primeira linha da matriz de dados Xtreino, resultando em Xtreino tendo ordem (p + 1) × N
# Xtreino = np.append(-np.ones((1, Xtreino.shape[1])), Xtreino, axis=0)

# TREINAMENTO


# forward
def forward(xamostra):
  # 2: j ←− 0
  j = 0
  # 3: for cada matriz de peso W em cada uma das L + 1 camadas. do
  for i in range(L + 1):
    # 4: if j == 0 then
    if j == 0:
      # 5: i[j] ←− W[j] · xamostra
      i[j] = W[j] @ xamostra
      # 6: y[j] ←− g(i[j])
      y[j] = g(i[j])
    # 7: else
    else:
      # 8: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
      ybias = np.append(-np.ones((1, y[j - 1].shape[1])), y[j - 1], axis=0)
      # 9: i[j] ←− W[j] · ybias
      i[j] = W[j] @ ybias
      # 10: y[j] ←− g(i[j])
      y[j] = g(i[j])
    # 11: end if
    # 12: j ←− j + 1
    j = j + 1
  # 13: end for

# backward
def backward(xamostra):
  # 2: j ←− Quantidade de matrizes W − 1.
  j = len(W) - 1
  # 3: while j ≥ 0 do
  while j >= 0:
    # 4: if j + 1 ==Quantidade de matrizes W, then
    if j + 1 == len(W):
      # 5: δ[j] ←− g
      # ′
      # (i[j]) ◦ (d − y[j]).
      delta[j] = g_(i[j]) * (d - y[j])
      # 6: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
      ybias = np.append(-np.ones((1, y[j - 1].shape[1])), y[j - 1], axis=0)
      # 7: W[j] ←− W[j] + η(δ[j] ⊗ ybias)
      W[j] = W[j] + n * (delta[j] @ ybias.T)
      # 8: else if j == 0 then
    elif j == 0:
      # 9: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
      Wb = W[j + 1].T[1:].T
      # 10: δ[j] ←− g
      # ′
      # (i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
      delta[j] = g_(i[j]) * (Wb @ delta[j + 1])
      # 11: W[j] ←− W[j] + η(δ[j] ⊗ xamostra)
      W[j] = W[j] + n * (delta[j] @ xamostra.T)
      # 12: else
    else:
      # 13: Wb[j + 1] Recebe a matriz W[j + 1] transposta sem a coluna que multiplica pelos limiares de ativação.
      Wb = W[j + 1].T[1:].T
      # 14: δ[j] ←− g′ (i[j]) ◦ (Wb[j + 1] · δ[j + 1]).
      delta[j] = g_(i[j]) * (Wb @ delta[j + 1])
      # 15: ybias ←− y[j − 1] com adição de −1 na primeira posição do vetor.
      ybias = np.append(-np.ones((1, y[j - 1].shape[1])), y[j - 1], axis=0)
      # 16: W[j] ←− W[j] + η(δ[j] ⊗ ybias)
      W[j] = W[j] + n * (delta[j] @ ybias.T)
    # 18: j ←− j − 1
    j = j - 1

# 1: EQM ←− 1.
eqm = 1
# 2: Epoch ←− 0.
epoch = 0
# 3: while EQM>CritérioParada && Epoch<MaxEpoch do
while(eqm>maxError and epoch<maxEpoch):
# 4: for Cada amostra em Xtreino do
    for i in range(Xtreino.shape[1]):
      # 5: xamostra ←− N−ésima amostra de Xtreino.
      xamostra = Xtreino[:, i]
      # 6: Forward(xamostra)
      forward(xamostra)
      # 7: d ←− N−ésimo rótulo de Xtreino.
      d = Ytreino[:, i]
      # 8: BackWard(xamostra, d).
      backward(xamostra, d)
      # 9: end for
    
    # 10: EQM ←− EQM().
    eqm = calcularEQM()
    # 11: Epoch ←−Epoch +1.
    epoch = epoch + 1
# 12: end while

# EQM
def calcularEQM():
  # 1: EQM ←− 0
  eqm = 0
  # 2: for Cada amostra em Xtreino do
  for i in range(Xtreino.shape[1]):
    # 3: xamostra ←− N−ésima amostra de Xtreino.
    xamostra = Xtreino[:, i]
    # 4: Forward(xamostra)
    forward(xamostra)
    # 5: d ←− N−ésimo rótulo de Xtreino.
    d = Ytreino[:, i]
    # 6: EQI ←− 0
    eqi = 0
    # 7: j ←− 0
    j = 0
    # 8: for Cada neurônio na camada de saída do
    for yl in y[-1]:
      # 9: EQI ←− EQI + (d[j] − y[QTD_L − 1][j])2
      eqi = eqi + (d[j] - yl) ** 2
      # 10: j ←− j + 1
      j = j + 1
    # 11: end for
    # 12: EQM ←− EQM + EQI
    eqm = eqm + eqi
  # 14: EQM ←− EQM/(2 ∗ QtdAmostrasTreino)
  eqm = eqm / (2 * Xtreino.shape[1])

# TESTE

# 1: for Cada amostra em Xteste do
for i in range(Xtreino.shape[1]):
  xamostra = Xtreino[:, i]
  forward(xamostra)


print("EQM: ", eqm)