import numpy as np
import matplotlib.pyplot as plt
from util import processar_dados, estimar_modelo

# Passo 1: Carregar os dados de um arquivo CSV
dados_aerogerador = np.genfromtxt("data/si/aerogerador.dat")

# Dividir os dados em variáveis independentes (X) e a variável de resposta (y)
X = dados_aerogerador[:, 0:1]
y = dados_aerogerador[:, 1].reshape(X.shape[0], 1)

for i in range(1000):
    # Passo 2: Preparar os dados
    # Embaralhe as amostras de X e Y
    # Divida X e Y em (X_treino, Y_treino) e (X_teste, Y_teste)
    (X_treino, y_treino, X_teste, y_teste, X_random, y_random) = processar_dados(X, y)

    # Passo 3: Implementar o modelo de regressão linear
    # Adicione um termo de viés (intercept) ao conjunto de dados
    # Calcular os coeficientes usando os Mínimos Quadrados Ordinários (MQO)
    (b_hat, X_treino) = estimar_modelo(X_treino, y_treino)

    # Passo 4: Fazer previsões
    # Faça as previsões usando o modelo treinado
    y_pred = X_treino @ b_hat


# Passo 5: Plotar os resultados
plt.scatter(X_random, y_random, color="k", label="Dados Originais")
plt.plot(X_treino[:, 1], y_pred[:, 0], color="red", label="Regressão Linear")
plt.show()
