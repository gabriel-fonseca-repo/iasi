# Trabalho computacional 2 (AV2)

1. Primeira etapa: problema de classificação binária com dados sintetizados aleatoriamente com a ferramenta `GerarDados_Parte1.py` que gerará dados com duas características $(X)$ e a primeira metade classificada como $1$, e a outra metade classificada como $-1$ $(Y)$. Os modelos implementados e utilizados para solucionar o problema de classificação são o `perceptron` e o `ADALINE`.

    - Implementar o modelo `perceptron`.
    - Implementar o modelo `ADALINE`.
    - Visualização inicial dos dados:
        - Gráfico de espalhamento para o conjunto de dados gerado.
        - Discutir quais resultados poderão ser obtidos ao utilizar os dois modelos.
    - Organizar o conjunto de dados de forma a obter $X\in\mathbb{R}^{(p+1)\times N}$.
    - Definir $\eta$, que se trata do passo de aprendizagem para o modelo `ADALINE`.
    - Definir o valor de precisão do `ADALINE`.
    - $100$ rodadas de treinamento (escolha do professor).
    - Divisão do conjunto de dados em $80/20$.
    - Computar as seguintes métricas para os resultados de cada modelo ao final das $100$ rodadas:
        - Acurácia -> média, desvio padrão, maior e menor.
        - Sensibilidade -> média, desvio padrão, maior e menor.
        - Especificidade -> média, desvio padrão, maior e menor.
    - Construir uma matriz de confusão para a pior e melhor rodada.
    - Exibir o hiperplano de separação dos dois modelos para a pior e melhor rodada.

<br>

2. Segunda etapa: $20$ classes, onde cada classe é referente a uma pessoa. É um problema de reconhecimento facial. Os modelos implementados para solucionar este problema são o `perceptron de múltiplas camadas (MLP)` e o `RBF`.

    - Implementar o modelo `MLP`.
    - Implementar o modelo `RBF`.
    - Realizar a normalização dos dados com base no método min-max (?).
    - Organizar o conjunto de dados de forma a obter $X\in\mathbb{R}^{(p+1)\times N}$.
    - Definir $\eta$, que se trata do passo de aprendizagem para ambos os modelos.
    - Para o modelo `MLP`:
        - Fazer uma discussão inicial sobre underfitting e overfitting.
        - Realizar um projeto de uma rede com poucos neurônios na camada oculta:
            - Computar valores de acurácia.
            - Computar matrizes de confusão.
        - Aumentar os neurônios e/ou camadas ocultas.
        - Identificar o overfitting:
            - Produzir duas matrizes de confusão para expressar os resultados.
        - Validar o modelo utilizando uma topologia da sua escolha:
            - Não deve produzir nem overfitting bem como underfitting.
            - Escolher com base nas instruções do professor (?).
    - Para o modelo `RBF`:
        - Escolher quantidade de funções de base.
    - $20$ rodadas de treinamento (escolha arbitrária).
    - Computar as seguintes métricas para os resultados de cada modelo ao final das $20$ rodadas:
        - Quantidade média de épocas para que os modelos atinjam convergência (?).
        - Acurácia -> média, desvio padrão, maior e menor.
        - Sensibilidade -> média, desvio padrão, maior e menor.
        - Especificidade -> média, desvio padrão, maior e menor.
    - Construir uma matriz de confusão para a pior e melhor rodada.
    - Projetar uma rede `MLP` que:
        - Seja superdimensionada.
        - Identifique o acontecimento de overfitting.
        - Faça uma parada antecipada ao identificar overfitting.
    - Realizar discussões se essa arquitetura obteve resultados melhores ou piores com relação às topologias anteriores e a rede `RBF`.
    - Teste