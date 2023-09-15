from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List
from modelos import (
    dmc,
    eqm_classificacao_ols,
    knn,
    mqo,
    mqo_sem_intercept_bidimensional,
    mqo_sem_intercept_tridimensional,
    mqo_tikhonov,
    media_b,
    media_b_tridimensional,
    eqm,
)
from util import (
    calcular_classes_preditoras,
    concatenar_uns,
    definir_melhor_lambda,
    processar_dados,
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
    EQM_OLS_S = []
    EQM_OLS_T = []
    melhor_lambda = None
    for _ in range(1000):
        (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)

        if not melhor_lambda:
            melhor_lambda = definir_melhor_lambda(
                X_treino, y_treino, X_teste, y_teste, lbds
            )
        b_media = media_b(y_treino)
        b_hat_ols_c = mqo(X_treino, y_treino)
        b_hat_ols_s = mqo_sem_intercept_bidimensional(X_treino, y_treino)
        b_hat_ols_t = mqo_tikhonov(X_treino, y_treino, melhor_lambda)

        X_teste = concatenar_uns(X_teste)

        y_pred_media = X_teste @ b_media
        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_s = X_teste @ b_hat_ols_s
        y_pred_ols_t = X_teste @ b_hat_ols_t

        EQM_MEDIA.append(eqm(y_teste, y_pred_media))
        EQM_OLS_C.append(eqm(y_teste, y_pred_ols_c))
        EQM_OLS_S.append(eqm(y_teste, y_pred_ols_s))
        EQM_OLS_T.append(eqm(y_teste, y_pred_ols_t))
    return (EQM_MEDIA, EQM_OLS_C, EQM_OLS_S, EQM_OLS_T)


def testar_eqm_modelos_regressao_tridimensional(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    lbds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
):
    EQM_MEDIA = []
    EQM_OLS_C = []
    EQM_OLS_S = []
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
        b_hat_ols_s = mqo_sem_intercept_tridimensional(X_treino, y_treino)
        b_hat_ols_t = mqo_tikhonov(X_treino, y_treino, melhor_lambda)

        X_teste = concatenar_uns(X_teste)

        y_pred_media = X_teste @ b_media
        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_s = X_teste @ b_hat_ols_s
        y_pred_ols_t = X_teste @ b_hat_ols_t

        EQM_MEDIA.append(eqm(y_teste, y_pred_media))
        EQM_OLS_C.append(eqm(y_teste, y_pred_ols_c))
        EQM_OLS_S.append(eqm(y_teste, y_pred_ols_s))
        EQM_OLS_T.append(eqm(y_teste, y_pred_ols_t))
    return (EQM_MEDIA, EQM_OLS_C, EQM_OLS_S, EQM_OLS_T)


def testar_eqm_modelos_classificacao(
    X: np.ndarray[Any, np.dtype[Any]],
    classes: List[str],
    lbds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
):
    EQM_OLS_C = []
    EQM_OLS_T = []
    EQM_OLS_K = []
    EQM_OLS_D = []
    melhor_lambda = None

    y = calcular_classes_preditoras(classes)

    for _ in range(1000):
        (X_treino, y_treino, X_teste, y_teste, _, _) = processar_dados(X, y)

        if not melhor_lambda:
            melhor_lambda = 1 or definir_melhor_lambda(
                X_treino, y_treino, X_teste, y_teste, lbds
            )
        b_hat_ols_c = 1 or mqo(X_treino, y_treino)
        b_hat_ols_t = 1 or mqo_tikhonov(X_treino, y_treino, melhor_lambda)
        # b_hat_ols_k = knn(X_treino, y_treino, X_teste, y_teste, classes)
        b_hat_ols_d = dmc(X_treino, y_treino, X_teste, y_teste, classes)

        X_teste = concatenar_uns(X_teste)

        y_pred_ols_c = X_teste @ b_hat_ols_c
        y_pred_ols_t = X_teste @ b_hat_ols_t

        # TODO: Implementar cálculo da taxa de acerto do KNN e do DMC

        EQM_OLS_C.append(eqm_classificacao_ols(y_teste, y_pred_ols_c))
        EQM_OLS_T.append(eqm_classificacao_ols(y_teste, y_pred_ols_t))
    return (EQM_OLS_C, EQM_OLS_T, EQM_OLS_K, EQM_OLS_D)


def boxplot_eqm(input: dict):
    plt.boxplot(
        input.values(),
        labels=input.keys(),
    )
    plt.show()


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


def visualizar_dados_sigmoidais(
    X: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]
):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], y[:, 0], color="orange", edgecolors="k")
    plt.show()
