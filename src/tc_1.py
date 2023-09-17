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

"""
  Para o trabalho:
  Passo 1 - FEITO! - Visualizar os dados
  Passo 2 - FEITO! - Organizar os dados (Separar variáveis regressoras e preditoras)
  Passo 3 - FEITO! - Definir quantidade de rodadas como 1000
  Passo 4 - FEITO! - Implementar os modelos MQO, MQO T, MVO
  Passo 5 - FEITO! - Determinar melhor valor de Lambda para MQO T
  Passo 6 - FEITO! - Embaralhar os dados e separá-los em conjunto de treino e conjunto de testes
  Passo 7 - FEITO! - Calcular o EQM para cada rodada de treino do modelo
  Passo 8 - FEITO! - Calcular a média, desvio-padrão, valor max e min de cada conjunto de EQM
"""

(X_aer, y_aer) = carregar_dados_aerogerador("data/aerogerador.dat")
(X_sig, y_sig) = carregar_dados_sigmoidais("data/DadosSigmoidais3d.csv")
X_EMG = carregar_dados_emg("data/EMG_Classes.csv")

# visualizar_dados_aerogerador(X_aer, y_aer)
# visualizar_dados_sigmoidais(X_sig, y_sig)
# visualizar_dados_emg(X_EMG)


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
            "OLS Tradicional": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )
    estatisticas_modelos_reg(EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


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
            "OLS Tradicional": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
        }
    )
    estatisticas_modelos_reg(EQM_MEDIA, EQM_OLS_C, EQM_OLS_T)


def resultado_classificacao_emg():
    (
        EQM_OLS_C,
        EQM_OLS_T,
        TAXA_ACERTO_KNN,
        TAXA_ACERTO_DMC,
    ) = testar_eqm_modelos_classificacao(
        X_EMG,
        classes=["Neutro", "Sorrindo", "Aberto", "Surpreso", "Rabugento"],
    )
    boxplot_eqm(
        {
            "OLS": EQM_OLS_C,
            "OLS Tikhonov (Regularizado)": EQM_OLS_T,
            "KNN": TAXA_ACERTO_KNN,
            "DMC": TAXA_ACERTO_DMC,
        }
    )
    estatisticas_modelos_classificacao(
        EQM_OLS_C, EQM_OLS_T, TAXA_ACERTO_KNN, TAXA_ACERTO_DMC
    )


resultado_regressao_aerogerador()
resultado_regressao_sigmoidais()
resultado_classificacao_emg()
