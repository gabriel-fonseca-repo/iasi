import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------LENDO O ARQUIVO-------------
arquivo_csv = "data/DataAV2_O.csv"
dados = np.loadtxt(arquivo_csv, delimiter=",")

X = dados[:, :-1]  # Todas as linhas, exceto a última coluna
y = dados[:, -1]  # Todas as linhas, apenas a última coluna


def plotar_grafico_de_dispersao(X, y, has_bias=True):
    if X.shape[0] == 2 or X.shape[0] == 3:
        X = X.T

    N, p = X.shape

    x1 = X[:, 1 if has_bias else 0].reshape(N, 1)
    x2 = X[:, 2 if has_bias else 1].reshape(N, 1)

    plt.scatter(x1[y == 1], x2[y == 1], c="blue", label="Classe 1", edgecolors="k")
    plt.scatter(x1[y == -1], x2[y == -1], c="red", label="Classe -1", edgecolors="k")

    plt.xlim(-12, 12)
    plt.ylim(-12, 12)

    plt.xlabel("Atributo 1")
    plt.ylabel("Atributo 2")
    plt.title("Gráfico de Dispersão de Amostras")
    plt.legend(loc="best")
    # plt.show()


plotar_grafico_de_dispersao(X, y, has_bias=False)

#################################
N, p = X.shape
X = X.T
y.shape = (len(y), 1)
X = np.concatenate((-np.ones((1, N)), X))
LR = 0.001
w = np.random.rand(p + 1, 1)
erro_existente = True

# ----------------Inicializando listas----------------
ACURACIA = []
SENSIBILIDADE = []
ESPECIFICIDADE = []

# ------------------------FUNÇÕES------------------------


# --------FUNÇÃO DE ATIVAÇÃO---------
def sign(u):
    if u >= 0:
        return 1
    else:
        return -1


# --------FUNÇÃO DE EMBARALHAR DADOS---------
def embaralhar_dados(X, y):
    X = X.T
    N, _ = X.shape
    seed = np.random.permutation(X.shape[0])
    X_random = X[seed, :]
    y_random = y[seed, :]
    X_treino = X_random[0 : int(N * 0.8), :]  # 0 até 80%
    y_treino = y_random[0 : int(N * 0.8), :]

    X_teste = X_random[int(N * 0.8) :, :]  # 80% até o final
    y_teste = y_random[int(N * 0.8) :, :]

    return (X_treino.T, y_treino.T, X_teste.T, y_teste.T)


# --------FUNÇÃO DE CALCULAR RESULTADOS---------
def calcula_resultados(matriz_confusao):
    verdadeiro_positivo: int = matriz_confusao[0, 0]
    falso_positivo: int = matriz_confusao[0, 1]
    falso_negativo: int = matriz_confusao[1, 0]
    verdadeiro_negativo: int = matriz_confusao[1, 1]

    acertos = verdadeiro_positivo + verdadeiro_negativo
    total = verdadeiro_positivo + verdadeiro_negativo + falso_negativo + falso_positivo

    acuracia = acertos / total
    sensibilidade = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo)
    especificidade = verdadeiro_negativo / (verdadeiro_negativo + falso_positivo)

    return (acuracia, sensibilidade, especificidade)


# --------FUNÇÃO DE CALCULAR MEDIDAS ESTATÍSTICAS---------
def calcula_medidas_estatisticas(ACURACIA, SENSIBILIDADE, ESPECIFICIDADE):
    stats = {
        "Medidas": ["Acurácia", "Sensibilidade", "Especificidade"],
        "Média": [np.mean(ACURACIA), np.mean(SENSIBILIDADE), np.mean(ESPECIFICIDADE)],
        "Desvio Padrão": [
            np.std(ACURACIA),
            np.std(SENSIBILIDADE),
            np.std(ESPECIFICIDADE),
        ],
        "Máximo": [np.max(ACURACIA), np.max(SENSIBILIDADE), np.max(ESPECIFICIDADE)],
        "Mínimo": [np.min(ACURACIA), np.min(SENSIBILIDADE), np.min(ESPECIFICIDADE)],
    }


# --------FUNÇÃO DE PREENCHER MATRIZ DE CONFUSÃO---------
def preenche_matriz_confusao(y_t, y_teste, i):
    indice_y_desejado = 1 if y_teste[i][0] == -1 else 0
    indice_y_predito = 1 if y_t == -1 else 0
    return (indice_y_desejado, indice_y_predito)


# ------------------------TREINAMENTO E TESTE------------------------
# Número de rodadas
num_rounds = 100

max_epoch = 10

for _ in range(num_rounds):
    (X_treino, y_treino, X_teste, y_teste) = embaralhar_dados(X, y)
    N = X_treino.shape[1]
    N_teste = X_teste.shape[1]

    x1 = np.linspace(-12, 12, N)
    x2 = np.zeros((N,))

    epoch = 0

    while erro_existente and epoch < max_epoch:
        erro_existente = False
        num_erros = 0
        matriz_confusao = np.zeros((2, 2))

        # ------------TREINO-------------
        for i in range(N):
            x_t = X_treino[:, i].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = y_treino[0, i]

            e_t = int(d_t - y_t)
            w = w + (e_t * x_t * LR) / 2

            if d_t != y_t:
                erro_existente = True
                num_erros += 1
                x2 = -x1 * (w[1, 0] / w[2, 0]) + w[0, 0] / w[2, 0]

        plt.plot(x1, x2, color=np.random.rand(3), alpha=0.4)
        plt.pause(0.01)
        # -----------TESTE--------------
        for i in range(N_teste):
            x_t = X_teste[:, i].reshape(3, 1)
            u_t = w.T @ x_t
            y_t = sign(u_t[0, 0])
            d_t = y_teste[0, i]
            verdadeiro, predito = preenche_matriz_confusao(y_t, y_teste.T, i)
            matriz_confusao[verdadeiro][predito] += 1
            # if i % 100 == 0:
            #     print(
            #         "Época: {0}, Amostra: {1}, y_t: {2}, y_desejado: {3}".format(
            #             epoch, i, y_t, d_t
            #         ))

        # -------------Medidas estatísticas--------------
        acuracia, sensibilidade, especificidade = calcula_resultados(matriz_confusao)
        ACURACIA.append(acuracia)
        SENSIBILIDADE.append(sensibilidade)
        ESPECIFICIDADE.append(especificidade)

        epoch += 1

    plt.cla()
    plotar_grafico_de_dispersao(X, y, has_bias=True)

calcula_medidas_estatisticas(ACURACIA, SENSIBILIDADE, ESPECIFICIDADE)
