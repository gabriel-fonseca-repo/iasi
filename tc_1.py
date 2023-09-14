from processo import (
    boxplot_eqm,
    testar_eqm_modelos_classificacao,
    testar_eqm_modelos_regressao,
)
from util import (
    carregar_dados_aerogerador,
    carregar_dados_emg,
    carregar_dados_sigmoidais,
)


(X_aer, y_aer) = carregar_dados_aerogerador("data/aerogerador.dat")
(X_sig, y_sig) = carregar_dados_sigmoidais("data/DadosSigmoidais3d.csv")
X_EMG = carregar_dados_emg("data/EMG.csv")


def resultado_regressao_aerogerador():
    (
        EQM_MEDIA,
        EQM_OLS_C,
        EQM_OLS_S,
        EQM_OLS_T,
    ) = testar_eqm_modelos_regressao(X=X_aer, y=y_aer)
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
    ) = testar_eqm_modelos_regressao(X=X_sig, y=y_sig)
    boxplot_eqm(
        {
            "Média": EQM_MEDIA,
            "OLS com i": EQM_OLS_C,
            "OLS sem i": EQM_OLS_S,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )


def resultado_classificacao_emg():
    (EQM_OLS_C, EQM_OLS_T, EQM_OLS_K, EQM_OLS_D) = testar_eqm_modelos_classificacao(
        X_EMG,
        classes=["Neutro", "Sorrindo", "Aberto", "Surpreso", "Rabugento"],
    )
    boxplot_eqm(
        {
            "OLS": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
            "KNN": EQM_OLS_K,
            "DMC": EQM_OLS_D,
        }
    )


# resultado_regressao_aerogerador()
# resultado_regressao_sigmoidais()
resultado_classificacao_emg()
