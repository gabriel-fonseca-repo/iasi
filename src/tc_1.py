from processo import (
    boxplot_eqm,
    estatisticas_modelos_classificacao,
    estatisticas_modelos_reg,
    testar_eqm_modelos_classificacao,
    testar_eqm_modelos_regressao,
    visualizar_dados_aerogerador,
    visualizar_dados_emg,
    visualizar_dados_sigmoidais,
)
from util import (
    carregar_dados_aerogerador,
    carregar_dados_emg,
    carregar_dados_sigmoidais,
)

import os

os.system("clear")

(X_aer, y_aer) = carregar_dados_aerogerador("data/aerogerador.dat")
(X_sig, y_sig) = carregar_dados_sigmoidais("data/DadosSigmoidais3d.csv")
X_EMG = carregar_dados_emg("data/EMG_Classes.csv")


def resultado_regressao_aerogerador():
    (
        EQM_MEDIA,
        EQM_OLS_C,
        EQM_OLS_T,
    ) = testar_eqm_modelos_regressao(X=X_aer, y=y_aer)
    boxplot_eqm(
        {
            "Média": EQM_MEDIA,
            "OLS Tradicional": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )
    estatisticas_modelos_reg(EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


def resultado_regressao_sigmoidais():
    (
        EQM_MEDIA,
        EQM_OLS_C,
        EQM_OLS_T,
    ) = testar_eqm_modelos_regressao(X=X_sig, y=y_sig)
    boxplot_eqm(
        {
            "Média": EQM_MEDIA,
            "OLS Tradicional": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )
    estatisticas_modelos_reg(EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


def resultado_classificacao_emg():
    (
        TAXA_ACERTO_OLS_C,
        TAXA_ACERTO_OLS_T,
        TAXA_ACERTO_KNN,
        TAXA_ACERTO_DMC,
    ) = testar_eqm_modelos_classificacao(
        X_EMG,
        classes=["Neutro", "Sorrindo", "Aberto", "Surpreso", "Rabugento"],
    )
    boxplot_eqm(
        {
            "OLS": TAXA_ACERTO_OLS_C,
            "OLS Tikhonov (Regularizado)": TAXA_ACERTO_OLS_T,
            "KNN": TAXA_ACERTO_KNN,
            "DMC": TAXA_ACERTO_DMC,
        }
    )
    estatisticas_modelos_classificacao(
        TAXA_ACERTO_OLS_C, TAXA_ACERTO_OLS_T, TAXA_ACERTO_KNN, TAXA_ACERTO_DMC
    )


visualizar_dados_aerogerador(X_aer, y_aer)
visualizar_dados_sigmoidais(X_sig, y_sig)
visualizar_dados_emg(X_EMG)

resultado_regressao_aerogerador()
resultado_regressao_sigmoidais()
resultado_classificacao_emg()
