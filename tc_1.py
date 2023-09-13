import numpy as np
import matplotlib.pyplot as plt
from util import testar_eqm_modelos

dados_aerogerador = np.genfromtxt("data/aerogerador.dat")

X_aerogerador = dados_aerogerador[:, 0:1]
y_aerogerador = dados_aerogerador[:, 1].reshape(X_aerogerador.shape[0], 1)


def resultado_regressao():
    (EQM_MEDIA, EQM_OLS_C, EQM_OLS_S, EQM_OLS_T) = testar_eqm_modelos(
        X_aerogerador, y_aerogerador
    )
    boxplot = [EQM_MEDIA, EQM_OLS_C, EQM_OLS_S, EQM_OLS_T]
    plt.boxplot(
        boxplot,
        labels=["Média", "OLS com i", "OLS sem i", "OLS Tikhonov (Regularizado)"],
    )
    plt.show()


def resultado_classificacao():
    pass


resultado_regressao()
resultado_classificacao()
