import random
import time
from math import sqrt

# 1. Fa¸ca a defini¸c˜ao de quantos pontos devem ser gerados por regi˜ao. Escolha um valor 30 < Npontos < 60.
n_pontos = 50  # Quantidade de pontos 30 < Npontos < 60

# 2. Faça a definição da quantidade N de indivíduos em uma população e quantidade m´axima de gera¸c˜oes.
qntd_individuos = 10  # Quantidade de indivíduos
max_geracoes = 1  # Quantidade máxima de gerações
probabilidade_recombinacao = 0.95  # Probabilidade de recombina¸c˜ao (85% a 95%)
probabilidade_mutacao = 0.01  # Probabilidade de muta¸c˜ao (1%)

individuos = [None] * qntd_individuos  # Array de indivíduos

# contador de tempo
tempo = time.time()


def gerar_gene():
    ...


class individuo:
    def __init__(self, cromossomo):
        if cromossomo == None:
            self.cromossomo = self.gerar_cromossomo()
        else:
            self.cromossomo = cromossomo
        self.calcular_aptidao()

    def calcular_aptidao(self):
        aptidao = 0
        ponto_atual = self.cromossomo[0]
        for i in range(n_pontos):
            aptidao += distancia_entre_pontos(ponto_atual, self.cromossomo[i])
            ponto_atual = self.cromossomo[i]
        self.aptidao = aptidao
        return aptidao

    def gerar_cromossomo(self):
        cromosomo = []
        for i in range(n_pontos):
            cromosomo.append(gerar_ponto())
        return cromosomo

    def get_range(self, posicao_inicial, posicao_final):
        return self.cromossomo[posicao_inicial:posicao_final]

    def tentar_mutacao(self):
        if probabilidade_mutacao():
            posicao = random.randint(0, 7)
            self.cromossomo[posicao] = gerar_gene()
            self.aptidao = self.calcular_aptidao()

    def toString(self):
        return str(self.cromossomo) + " - " + str(self.aptidao)


def gerar_ponto():
    x = random.randint(0, 30)
    y = random.randint(0, 30)
    z = random.randint(0, 30)
    return [x, y, z]


def distancia_entre_pontos(ponto1, ponto2):
    return sqrt(
        (ponto1[0] - ponto2[0]) ** 2
        + (ponto1[1] - ponto2[1]) ** 2
        + (ponto1[2] - ponto2[2]) ** 2
    )


def roleta(aptidao, individuos):
    # invertendo aptidao para que o maior valor seja o menor e vice-versa
    soma = 0
    for i in range(len(aptidao)):
        aptidao[i] = aptidao[i] * -1
        soma += aptidao[i]

    for i in range(len(aptidao)):
        aptidao[i] = aptidao[i] / soma

    pais = [None] * 2
    for i in range(2):
        valor = random.random()
        for j in range(len(aptidao)):
            valor -= aptidao[j]
            if valor <= 0:
                pais[i] = individuos[j]
                break
    return pais


for i in range(qntd_individuos):
    individuos[i] = individuo(None)

geracao_atual = 0
while geracao_atual < max_geracoes:
    aptidao = [None] * qntd_individuos

    for i in range(qntd_individuos):
        aptidao[i] = individuos[i].aptidao

    pais = [None] * 2

    # 3. Projete o operador de seleção, baseado no método do torneio.
    pais = roleta(aptidao, individuos)

    # 4. Na etapa de recombinação, como este trata-se de um problema de combinatória, não pode haver pontos repetidos na sequência cromossômica. Desta maneira, pede-se que desenvolva uma variação do operador de recombina¸c˜ao de dois pontos. Assim, cada seção selecionada de modo aleatório deve ser propagada nos filhos e em seguida, a sequência genética do filho deve ser completada com os demais pontos sem repetição.

    print("---------------------------------------------------")
    print("Geração: ", geracao_atual)
    print("---------------------------------------------------")
    geracao_atual += 1

# 5. Na prole gerada, deve-se aplicar a muta¸c˜ao com probabilidade de 1%. Para o problema do caixeiro viajante, deve-se aplicar uma muta¸c˜ao que faz a troca de um gene por outro de uma mesma sequˆencia cromossˆomica.
# 6. O algoritmo deve parar quando atingir o m´aximo n´umero de gera¸c˜oes ou quando a fun¸c˜ao custo atingir seu valor ´otimo aceit´avel (de acordo com a regra descrita no slide 31/61).
# 7. Fa¸ca uma an´alise se de qual ´e a moda de gera¸c˜oes para obter uma solu¸c˜ao aceit´avel. Al´em disso, analise se ´e necess´ario incluir um operador de elitismo em um grupo Ne de indiv´ıduos.
