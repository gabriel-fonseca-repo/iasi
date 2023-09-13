import numpy as np
import matplotlib.pyplot as plt
from util import (
    processar_dados,
    concatenar_uns,
    estimar_modelo_ones,
    estimar_modelo_zeros,
    estimar_modelo_tikhonov,
    definir_melhor_lambda,
    eqm,
    media_b,
)

# Passo 1: Carregar os dados de um arquivo CSV
dados_aerogerador = np.genfromtxt("data/si/aerogerador.dat")

# Dividir os dados em variáveis independentes (X) e a variável de resposta (y)
X = dados_aerogerador[:, 0:1]
y = dados_aerogerador[:, 1].reshape(X.shape[0], 1)

EQM_MEDIA = []
EQM_OLS_C = []
EQM_OLS_S = []
EQM_OLS_T = []

lbds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]

melhor_lambda = None
for i in range(1000):
    # Passo 2: Preparar os dados
    # Embaralhe as amostras de X e Y
    # Divida X e Y em (X_treino, Y_treino) e (X_teste, Y_teste)
    (X_treino, y_treino, X_teste, y_teste, X_random, y_random) = processar_dados(X, y)

    # Passo 3: Implementar o modelo de regressão linear
    # Adicione um termo de viés (intercept) ao conjunto de dados
    # Calcular os coeficientes usando os Mínimos Quadrados Ordinários (MQO)
    if not melhor_lambda:
        melhor_lambda = definir_melhor_lambda(X_treino, y_treino, X_teste, y_teste, lbds)
    b_media = media_b(y_treino)
    b_hat_ols_c = estimar_modelo_ones(X_treino, y_treino)
    b_hat_ols_s = estimar_modelo_zeros(X_treino, y_treino)
    b_hat_ols_t = estimar_modelo_tikhonov(
        X_treino, y_treino, melhor_lambda
    )

    # Passo 4: Fazer previsões
    # Faça as previsões usando o modelo treinado
    X_teste = concatenar_uns(X_teste)

    y_pred_media = X_teste @ b_media
    y_pred_ols_c = X_teste @ b_hat_ols_c
    y_pred_ols_s = X_teste @ b_hat_ols_s
    y_pred_ols_t = X_teste @ b_hat_ols_t

    EQM_MEDIA.append(eqm(y_teste, y_pred_media))
    EQM_OLS_C.append(eqm(y_teste, y_pred_ols_c))
    EQM_OLS_S.append(eqm(y_teste, y_pred_ols_s))
    EQM_OLS_T.append(eqm(y_teste, y_pred_ols_t))


# Passo 5: Plotar os resultados
boxplot = [EQM_MEDIA, EQM_OLS_C, EQM_OLS_S, EQM_OLS_T]
plt.boxplot(boxplot, labels=["Média", "OLS com i", "OLS sem i", "OLS Tikhonov (Regularizado)"])
plt.show()