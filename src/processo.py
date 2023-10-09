from typing import Any, List

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from modelos import (
    dmc,
    eqm,
    knn,
    media_b,
    media_b_tridimensional,
    mqo,
    mqo_tikhonov,
    printar_progresso,
    ta_dmc,
    ta_knn,
    ta_ols,
)
from util import (
    calcular_classes_preditoras,
    concatenar_uns,
    definir_melhor_lambda,
    processar_dados,
    separar_classes,
)


def testar_eqm_modelos_regressao(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    p = X.shape[1]
    if p == 1:
        return testar_eqm_modelos_regressao_bidimensional(X, y)
    elif p == 2:
        return testar_eqm_modelos_regressao_tridimensional(X, y)
    else:
        raise Exception(
            f"Impossível discernir um cálculo para a ordem da matriz X: {X.shape}."
        )


def testar_eqm_modelos_regressao_bidimensional(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    lbds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
):
    EQM_MEDIA = []
    EQM_OLS_C = []
    EQM_OLS_T = []
    melhor_lambda = None
    for _ in range(1000):
        (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)

        if melhor_lambda is None:
            melhor_lambda = definir_melhor_lambda(
                X_treino, y_treino, X_teste, y_teste, lbds
            )
        b_media = media_b(y_treino)
        b_hat_ols_c = mqo(X_treino, y_treino)
        b_hat_ols_t = mqo_tikhonov(X_treino, y_treino, melhor_lambda)

        X_teste = concatenar_uns(X_teste)

        y_pred_media = X_teste @ b_media
        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_t = X_teste @ b_hat_ols_t

        EQM_MEDIA.append(eqm(y_teste, y_pred_media))
        EQM_OLS_C.append(eqm(y_teste, y_pred_ols_c))
        EQM_OLS_T.append(eqm(y_teste, y_pred_ols_t))

    return (EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


def testar_eqm_modelos_regressao_tridimensional(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    lbds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
):
    EQM_MEDIA = []
    EQM_OLS_C = []
    EQM_OLS_T = []
    melhor_lambda = None
    for i in range(1000):
        (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)

        if not melhor_lambda:
            melhor_lambda = definir_melhor_lambda(
                X_treino, y_treino, X_teste, y_teste, lbds
            )
        b_media = media_b_tridimensional(y_treino)
        b_hat_ols_c = mqo(X_treino, y_treino)
        b_hat_ols_t = mqo_tikhonov(X_treino, y_treino, melhor_lambda)

        X_teste = concatenar_uns(X_teste)

        y_pred_media = X_teste @ b_media
        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_t = X_teste @ b_hat_ols_t

        EQM_MEDIA.append(eqm(y_teste, y_pred_media))
        EQM_OLS_C.append(eqm(y_teste, y_pred_ols_c))
        EQM_OLS_T.append(eqm(y_teste, y_pred_ols_t))
    # plotar_grafico_tridimensional(X_treino, y_treino, b_hat_ols_c)
    return (EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


def testar_eqm_modelos_classificacao(
    X: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
    lbds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
):
    TAXA_ACERTO_OLS_C = []
    TAXA_ACERTO_OLS_T = []
    TAXA_ACERTO_KNN = []
    TAXA_ACERTO_DMC = []
    melhor_lambda = 0.1

    y = calcular_classes_preditoras(classes)

    qtd_rodadas = 100

    for rodada in range(qtd_rodadas):
        (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)
        (X_treino_sc, X_teste_sc) = separar_classes(X_treino, X_teste)

        if melhor_lambda is None:
            melhor_lambda = definir_melhor_lambda(
                X_treino_sc, y_treino, X_teste_sc, y_teste, lbds
            )
        b_hat_ols_c = mqo(X_treino_sc, y_treino)
        b_hat_ols_t = mqo_tikhonov(X_treino_sc, y_treino, melhor_lambda)
        predicoes_knn = knn(X_treino_sc, y_treino, X_teste_sc, y_teste, classes)
        predicoes_dmc = dmc(X_treino, y_treino, classes)

        X_teste = concatenar_uns(X_teste)

        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_t = X_teste @ b_hat_ols_t

        TAXA_ACERTO_OLS_C.append(ta_ols(y_teste, y_pred_ols_c) * 100)
        TAXA_ACERTO_OLS_T.append(ta_ols(y_teste, y_pred_ols_t) * 100)
        TAXA_ACERTO_KNN.append(ta_knn(predicoes_knn, X_teste.shape[0]) * 100)
        TAXA_ACERTO_DMC.append(ta_dmc(X_treino, predicoes_dmc) * 100)
        printar_progresso(rodada / qtd_rodadas)
    return (TAXA_ACERTO_OLS_C, TAXA_ACERTO_OLS_T, TAXA_ACERTO_KNN, TAXA_ACERTO_DMC)


def boxplot_eqm(input: dict):
    plt.boxplot(
        input.values(),
        labels=input.keys(),
    )
    plt.savefig("out/Estatisticas_ols.png")


def visualizar_dados_aerogerador(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    plt.scatter(X, y, color="blue", edgecolors="black")
    plt.show()


def visualizar_dados_emg(X: np.ndarray[Any, np.dtype[Any]]):
    colors = ["red", "green", "purple", "blue", "gray"]
    k = 0
    for _ in range(10):
        for color in colors:
            plt.scatter(
                X[k : k + 1000, 0],
                X[k : k + 1000, 1],
                color=color,
                edgecolors="k",
            )
            k += 1000
    plt.show()


def estatisticas_modelos_reg(EQM_MEDIA, EQM_OLS_C, EQM_OLS_T):
    stats = {
        "Modelo": ["Médias Observáveis", "OLS Tradicional", "Tikhonov"],
        "Média": [
            np.mean(EQM_MEDIA),
            np.mean(EQM_OLS_C),
            np.mean(EQM_OLS_T),
        ],
        "Desvio Padrão": [
            np.std(EQM_MEDIA),
            np.std(EQM_OLS_C),
            np.std(EQM_OLS_T),
        ],
        "Máximo": [
            np.max(EQM_MEDIA),
            np.max(EQM_OLS_C),
            np.max(EQM_OLS_T),
        ],
        "Mínimo": [
            np.min(EQM_MEDIA),
            np.min(EQM_OLS_C),
            np.min(EQM_OLS_T),
        ],
    }

    df = pd.DataFrame(stats)
    df.to_csv("out/rst_regressao.csv", sep=";")
    plt.figure(figsize=(10, 6))
    plt.bar(df["Modelo"], df["Média"], yerr=df["Desvio Padrão"])
    plt.xlabel("Modelo")
    plt.ylabel("Média de EQM")
    plt.title("Estatísticas de performance por Modelo")
    plt.savefig("out/Estatisticas_ols.png")


def visualizar_dados_sigmoidais(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], y[:, 0], color="orange", edgecolors="k")
    plt.show()


def plotar_grafico_tridimensional(X_treino, y_treino, b_hat_ols_c):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        X_treino[:, 1],
        y_treino[:, 0],
        color="purple",
        edgecolors="k",
    )
    x_axis = np.linspace(0, 40, 500)
    y_axis = np.linspace(0, 750, 500)

    X_map, Y_map = np.meshgrid(x_axis, y_axis)
    X_map.shape = (500, 500, 1)
    Y_map.shape = (500, 500, 1)
    ones_map = np.ones((500, 500, 1))
    X3D = np.concatenate((ones_map, X_map, Y_map), axis=2)
    Z_media = X3D @ b_hat_ols_c

    ax.plot_surface(
        X_map[:, :, 0], Y_map[:, :, 0], Z_media[:, :, 0], cmap="gray", alpha=0.5
    )
    plt.show()


def estatisticas_modelos_classificacao(
    TAXA_ACERTO_OLS_C,
    TAXA_ACERTO_OLS_T,
    TAXA_ACERTO_KNN,
    TAXA_ACERTO_DMC,
):
    MODA_TA_OLS_C = st.mode(TAXA_ACERTO_OLS_C)
    MODA_TA_OLS_T = st.mode(TAXA_ACERTO_OLS_T)
    MODA_TA_OLS_KNN = st.mode(TAXA_ACERTO_KNN)
    MODA_TA_OLS_DMC = st.mode(TAXA_ACERTO_DMC)
    stats = {
        "Modelo": ["OLS Tradicional", "OLS Tikhonov (Regularizado)", "K-NN", "DMC"],
        "Média": [
            np.mean(TAXA_ACERTO_OLS_C),
            np.mean(TAXA_ACERTO_OLS_T),
            np.mean(TAXA_ACERTO_KNN),
            np.mean(TAXA_ACERTO_DMC),
        ],
        "Desvio Padrão": [
            np.std(TAXA_ACERTO_OLS_C),
            np.std(TAXA_ACERTO_OLS_T),
            np.std(TAXA_ACERTO_KNN),
            np.std(TAXA_ACERTO_DMC),
        ],
        "Máximo": [
            np.max(TAXA_ACERTO_OLS_C),
            np.max(TAXA_ACERTO_OLS_T),
            np.max(TAXA_ACERTO_KNN),
            np.max(TAXA_ACERTO_DMC),
        ],
        "Mínimo": [
            np.min(TAXA_ACERTO_OLS_C),
            np.min(TAXA_ACERTO_OLS_T),
            np.min(TAXA_ACERTO_KNN),
            np.min(TAXA_ACERTO_DMC),
        ],
        "Moda": [
            str(MODA_TA_OLS_C[0]) + " (" + str(MODA_TA_OLS_C[1]) + ")",
            str(MODA_TA_OLS_T[0]) + " (" + str(MODA_TA_OLS_T[1]) + ")",
            str(MODA_TA_OLS_KNN[0]) + " (" + str(MODA_TA_OLS_KNN[1]) + ")",
            str(MODA_TA_OLS_DMC[0]) + " (" + str(MODA_TA_OLS_DMC[1]) + ")",
        ],
    }

    df = pd.DataFrame(stats)
    df.to_csv("out/rst_classificacao.csv", sep=";")
    plt.figure(figsize=(10, 6))
    plt.bar(df["Modelo"], df["Média"], yerr=df["Desvio Padrão"])
    plt.xlabel("Modelo")
    plt.ylabel("Média de TA")
    plt.title("Estatísticas de performance por Modelo")
    plt.savefig("out/Estatisticas_emg.png")
