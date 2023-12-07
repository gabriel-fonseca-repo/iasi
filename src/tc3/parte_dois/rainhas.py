import random
import time
from math import ceil
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# 1. Faça a definição da quantidade N de indivíduos em uma população e quantidade máxima de geraçõoes.
N = 100  # Quantidade de indivíduos
max_geracoes = 100000000  # Quantidade máxima de gerações

probabilidade_recombinacao = 0.95  # Probabilidade de recombinação (85% a 95%)
probabilidade_mutacao = 0.01  # Probabilidade de mutação (1%)

encontrar_todas_solucoes = True
total_solucoes = 92
solucoes_encontradas = 0
todas_solucoes = [None] * total_solucoes

individuos = [None] * N  # Array de indivíduos


# contador de tempo
tempo = time.time()


class individuo:
    def __init__(self, cromossomo):
        if cromossomo == None:
            self.cromossomo = self.gerar_cromossomo()
        else:
            self.cromossomo = cromossomo
        self.aptidao = self.calcular_aptidao()

    def calcular_aptidao(self):
        aptidao = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if self.cromossomo[i] == self.cromossomo[j]:
                    aptidao += 1
                elif self.cromossomo[i] == self.cromossomo[j] - (j - i):
                    aptidao += 1
                elif self.cromossomo[i] == self.cromossomo[j] + (j - i):
                    aptidao += 1

        return 28 - aptidao

    def gerar_cromossomo(self):
        cromosomo = []
        for i in range(8):
            cromosomo.append(gerar_gene())
        return cromosomo

    def get_range(self, posicao_inicial, posicao_final):
        return self.cromossomo[posicao_inicial:posicao_final]

    def tentar_mutacao(self):
        if calc_probabilidade_mutacao():
            posicao = random.randint(0, 7)
            self.cromossomo[posicao] = gerar_gene()
            self.aptidao = self.calcular_aptidao()

    def toString(self):
        return str(self.cromossomo)

    def print(self):
        print("Cromossomo: ", self.cromossomo)
        print("Aptidao: ", self.aptidao)


def roleta(aptidao, individuos):
    total_aptidao = sum(aptidao)
    aptidao_normalizada = [None] * N

    for j in range(N):
        aptidao_normalizada[j] = aptidao[j] / total_aptidao

    pais = [None] * 2

    for j in range(2):
        r = random.random()
        for k in range(N):
            r -= aptidao_normalizada[k]
            if r <= 0:
                pais[j] = individuos[k]
                break

    return pais


def recombinacao_um_ponto(pai, pai2):
    if not calc_probabilidade_recombinacao():
        return False
    posicao = random.randint(1, 7)

    filhos = [None] * 2

    filhos[0] = individuo(pai.get_range(0, posicao) + pai2.get_range(posicao, 8))
    filhos[1] = individuo(pai2.get_range(0, posicao) + pai.get_range(posicao, 8))

    return filhos


def recombinacao_dois_pontos(pai, pai2):
    if not calc_probabilidade_recombinacao():
        return False
    posicao1 = random.randint(1, 7)
    posicao2 = random.randint(1, 7)

    if posicao1 > posicao2:
        posicao1, posicao2 = posicao2, posicao1

    filhos = [None] * 2

    filhos[0] = individuo(
        pai.get_range(0, posicao1)
        + pai2.get_range(posicao1, posicao2)
        + pai.get_range(posicao2, 8)
    )
    filhos[1] = individuo(
        pai2.get_range(0, posicao1)
        + pai.get_range(posicao1, posicao2)
        + pai2.get_range(posicao2, 8)
    )

    return filhos


def calc_probabilidade_recombinacao():
    rand = random.random()
    if rand <= probabilidade_recombinacao:
        return True
    else:
        return False


def calc_probabilidade_mutacao():
    rand = random.random()
    if rand <= probabilidade_mutacao:
        return True
    else:
        return False


# gerar um gene aleatório (0 a 7)
def gerar_gene():
    return random.randint(0, 7)


for i in range(N):
    individuos[i] = individuo(None)

geracao_atual = 0
while geracao_atual < max_geracoes:
    aptidao = [None] * N

    terminou = False

    for i in range(N):
        aptidao[i] = individuos[i].aptidao
        if aptidao[i] == 28:
            if not encontrar_todas_solucoes:
                print("Solução encontrada!")
                individuos[i].print()
                terminou = True
            else:
                if individuos[i].toString() not in todas_solucoes:
                    # se for o primeiro, mostra graficamente o tabuleiro
                    if solucoes_encontradas == 0:
                        individuos[i].print()
                        # scaterplot 8x8 como tabuleiro de xadrez
                        chessboard = np.zeros((8, 8))
                        for i in range(8):
                            for j in range(8):
                                if (i + j) % 2 == 0:
                                    chessboard[i, j] = 1

                        plt.title("Solução ótima encontrada")
                        plt.imshow(chessboard, cmap="binary")
                        plt.xticks(range(1, 9))
                        plt.yticks(range(1, 9))
                        plt.scatter(
                            individuos[i].cromossomo,
                            range(1, 9),
                            color="red",
                            s=1000,
                            marker="s",
                        )
                        plt.show()

                    solucoes_encontradas += 1
                    todas_solucoes[solucoes_encontradas - 1] = individuos[i].toString()
                    if solucoes_encontradas == total_solucoes:
                        print("Todas as Soluções encontradas!")
                        print("Geracao: ", geracao_atual)
                        terminou = True
                        break

    # 5. O algoritmo deve parar quando atingir o máximo número de gerações ou quando a função custo atingir seu valor ótimo.
    if terminou:
        break

    # 2. Projete o operador de seleção, baseado no método da roleta.
    pais = roleta(aptidao, individuos)

    # 3. Na etapa de recombinação, escolha um entre os seguintes operadores: Recombinação de um ponto ou Recombinação de dois pontos. A probabilidade de recombinação nesta etapa deve ser entre 85 < pc < 95%.
    # fiz as duas opções para testar
    # recombinou = recombinacao_um_ponto(pais[0], pais[1])
    recombinou = recombinacao_dois_pontos(pais[0], pais[1])

    if recombinou:
        for i in range(N):
            if individuos[i] == pais[0]:
                individuos[i] = recombinou[0]
            elif individuos[i] == pais[1]:
                individuos[i] = recombinou[1]

    # 4. Na prole gerada, deve-se aplicar a mutação com probabilidade de 1% (neste caso, é interessante avaliar os diferentes procedimentos exibidos).
    for i in range(N):
        individuos[i].tentar_mutacao()

    geracao_atual += 1
    if geracao_atual % 100000 == 0:
        print("---------------------------------------------------")
        print("Geracao: ", geracao_atual)
        print("aptidao maxima: ", max(aptidao))
        if encontrar_todas_solucoes:
            print("Soluções encontradas: ", solucoes_encontradas)
            if geracao_atual == max_geracoes:
                if solucoes_encontradas == 0:
                    print("Nenhuma solução encontrada")
        print("---------------------------------------------------")


# tempo em horas, minutos e segundos
tempo_normal = time.time() - tempo
tempo_horas = ceil(tempo_normal // 3600)
tempo_normal %= 3600
tempo_minutos = ceil(tempo_normal // 60)
tempo_normal %= 60
tempo_segundos = ceil(tempo_normal)

print(
    "Tempo de execução: ", tempo_horas, "h ", tempo_minutos, "m ", tempo_segundos, "s"
)

# De posse do primeiro resultado, aplique o seu projeto de algoritmo genético para executar enquanto as 92 soluções diferentes não forem encontradas. Avalie o custo computacional desta etapa.
