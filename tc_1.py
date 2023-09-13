import numpy as np
import matplotlib.pyplot as plt
from processo import (
    boxplot_eqm,
    testar_eqm_modelos_classificacao,
    testar_eqm_modelos_regressao,
)
from util import (
    carregar_dados_bidimensionais,
    carregar_dados_tridimensionais,
)


(X_aer, y_aer) = carregar_dados_bidimensionais("data/aerogerador.dat")
(X_EMG, y_EMG) = carregar_dados_bidimensionais("data/EMG.csv")
(X_sig, y_sig) = carregar_dados_tridimensionais("data/DadosSigmoidais3d.csv")


def resultado_regressao_aerogerador():
    (
        EQM_MEDIA,
        EQM_OLS_C,
        EQM_OLS_S,
        EQM_OLS_T,
    ) = testar_eqm_modelos_regressao(X=X_aer, y=y_aer, ordem=2)
    boxplot_eqm(
        {
            "Média": EQM_MEDIA,
            "OLS com i": EQM_OLS_C,
            "OLS sem i": EQM_OLS_S,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )


def resultado_regressao_sigmoidais():
    (
        EQM_MEDIA,
        EQM_OLS_C,
        EQM_OLS_S,
        EQM_OLS_T,
    ) = testar_eqm_modelos_regressao(X=X_sig, y=y_sig, ordem=3)
    boxplot_eqm(
        {
            "Média": EQM_MEDIA,
            "OLS com i": EQM_OLS_C,
            "OLS sem i": EQM_OLS_S,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )


def resultado_classificacao_emg():
    (EQM_OLS_C, EQM_OLS_T) = testar_eqm_modelos_classificacao(X_EMG, y_EMG)
    boxplot_eqm(
        {
            "OLS": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_C,
            "KNN": EQM_OLS_T,
            "DMC": EQM_OLS_T,
        }
    )


# resultado_regressao_aerogerador()
resultado_regressao_sigmoidais()
# resultado_classificacao_emg()
