from processo import (
    boxplot_eqm,
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

"""
  Para o trabalho:
  Passo 1 - FEITO! - Visualizar os dados
  Passo 2 - FEITO! - Organizar os dados (Separar variáveis regressoras e preditoras)
  Passo 3 - FEITO! - Definir quantidade de rodadas como 1000
  Passo 4 - FEITO! - Implementar os modelos MQO, MQO T, MVO
  Passo 5 - FEITO! - Determinar melhor valor de Lambda para MQO T
  Passo 6 - FEITO! - Embaralhar os dados e separá-los em conjunto de treino e conjunto de testes
  Passo 7 - FEITO! - Calcular o EQM para cada rodada de treino do modelo
  Passo 8 - FALTA FAZER X - Calcular a média, desvio-padrão, valor max e min de cada conjunto de EQM

  FALTA FAZER - Passo 4 de classificação
"""

(X_aer, y_aer) = carregar_dados_aerogerador("data/aerogerador.dat")
(X_sig, y_sig) = carregar_dados_sigmoidais("data/DadosSigmoidais3d.csv")
X_EMG = carregar_dados_emg("data/EMG.csv")

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
