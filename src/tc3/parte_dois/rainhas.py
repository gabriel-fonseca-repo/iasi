import random
import string
import time
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

# contador de tempo
tempo = time.time()


def plotar_solucao_encontrada(atual, index_solucao):
    for i in range(8):
        atual[i] = 7 - atual[i]

    # Scatter novo:
    chessboard = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                chessboard[i, j] = 1

    chess_labels = list(
        string.ascii_uppercase[:8]
    )  # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    plt.title("Solução ótima encontrada:\n" + str(atual))
    plt.imshow(chessboard, cmap="binary")
    plt.scatter(
        range(8),
        atual,
        color="red",
        s=800,
    )
    plt.xticks(range(8), chess_labels)
    plt.yticks(range(8), [8, 7, 6, 5, 4, 3, 2, 1])
    plt.savefig(
        f"out/tc3/solucoes_oito_rainhas/Solucao_{index_solucao}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()


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

class populacao:
    def __init__(self, individuos):
        self.individuos = individuos

    def inicializar(self, tamanho):
        for i in range(tamanho):
            self.add_individuo(individuo(None))

    def add_individuo(self, individuo):
        self.individuos.append(individuo)

    def get_individuo(self, posicao):
        return self.individuos[posicao]
    
    def length(self):
        if self.individuos == None:
            return 0
        return len(self.individuos)
    
    def total_aptidao(self):
        total = 0
        for i in range(len(self.individuos)):
            total += self.individuos[i].aptidao
        return total

    def roleta(self):
        if self.length() == 0:
            return None
        
        if self.length() == 1:
            return self.individuos[0]
        
        aptidao = [None] * self.length()
        for i in range(self.length()):
            aptidao[i] = self.individuos[i].aptidao

        # seleciona um individuo aleatorio pelo metodo da roleta
        r = random.random()
        total_aptidao = sum(aptidao)
        aptidao_normalizada = [aptidao[i] / total_aptidao for i in range(self.length())]
        for i in range(self.length()):
            r -= aptidao_normalizada[i]
            if r <= 0:
                # remove o individuo selecionado
                temp = self.individuos[i]
                self.individuos.pop(i)
                return temp
            
    def has(self, individuo):
        for i in range(len(self.individuos)):
            cromossomoAtual = self.individuos[i].cromossomo
            flag = True
            for j in range(8):
                if individuo[j] != cromossomoAtual[j]:
                    flag = False
                    break
            if flag:
                return True
        return False
            
    def tentar_mutacao(self):
        for i in range(len(self.individuos)):
            self.individuos[i].tentar_mutacao()
    
    def checar_resultado(self):
        for i in range(len(self.individuos)):
            if self.individuos[i].aptidao == 28:
                return self.individuos[i].cromossomo
        return None


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
        return [pai, pai2]
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

def jaAchou(solucao):
    for i in range(total_solucoes):
        if todas_solucoes[i] != None:
            flag = True
            for j in range(8):
                if solucao[j] != todas_solucoes[i][j]:
                    flag = False
                    break
            if flag:
                return True
    return False

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
                atual2 = individuos[i].cromossomo.copy()
                if not jaAchou(individuos[i].cromossomo):
                    # se for o primeiro, mostra graficamente o tabuleiro
                    atualOriginal = individuos[i].cromossomo
                    # if solucoes_encontradas == 0:
                    #     individuos[i].print()
                    #     atual = individuos[i].cromossomo
                        
                    #     for i in range(8):
                    #         atual[i] = 7 - atual[i]

                    #     # Scatter novo:
                    #     chessboard = np.zeros((8, 8))
                    #     for i in range(8):
                    #         for j in range(8):
                    #             if (i + j) % 2 == 0:
                    #                 chessboard[i, j] = 1

                    #     chess_labels = list(
                    #         string.ascii_uppercase[:8]
                    #     )  # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

                    #     plt.title("Solução encontrada")
                    #     plt.imshow(chessboard, cmap="binary")
                    #     plt.scatter(
                    #         range(8),
                    #         atual,
                    #         color="red",
                    #         s=800,
                    #     )
                    #     plt.xticks(range(8), chess_labels)
                    #     plt.yticks(range(8), [8, 7, 6, 5, 4, 3, 2, 1])
                    #     plt.show()
                    solucoes_encontradas += 1
                    print("GUARDANDO SOLUÇÃO " + str(atualOriginal))
                    todas_solucoes[solucoes_encontradas - 1] = atual2
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

        resultado = individuos.checar_resultado()
        if(resultado != None):
            print("geração: ", geracao_atual)
            return resultado
        geracao_atual += 1
        

    print("não encontrou solução e chegou no limite de gerações")
    return None

def mostrar_grafico(cromossomo):
    for i in range(8):
        cromossomo[i] = 7 - cromossomo[i]

    # Scatter novo:
    chessboard = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                chessboard[i, j] = 1

    chess_labels = list(
        string.ascii_uppercase[:8]
    )  # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    plt.imshow(chessboard, cmap="binary")
    plt.scatter(
        range(8),
        cromossomo,
        color="red",
        s=800,
    )
    plt.xticks(range(8), chess_labels)
    plt.yticks(range(8), [8, 7, 6, 5, 4, 3, 2, 1])
    plt.show()


resultadosOtimos = populacao([])
universos = 0
while(resultadosOtimos.length() < total_solucoes):
    print("[", resultadosOtimos.length(),"/", total_solucoes ,"]" ,"---iniciando Universo Nº", universos + 1, "---")

    individuos = populacao([])
    individuos.inicializar(N)

    resultado = rodar_ate_encontrar(individuos)

    if(resultado != None):
        if(resultadosOtimos.has(resultado)):
            print("Solução já encontrada")
        else:
            resultadosOtimos.add_individuo(individuo(resultado))
            print("Solução nova encontrada " + individuo(resultado).toString())
            if(resultadosOtimos.length() == 0):
                mostrar_grafico(resultado)
    

    universos += 1
        


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

print("Quantidade de universos: ", universos)
print("Resultados ótimos encontrados: ", resultadosOtimos.length())
print("Resultados ótimos: ")
for i in range(resultadosOtimos.length()):
    print(resultadosOtimos.get_individuo(i).toString())

# De posse do primeiro resultado, aplique o seu projeto de algoritmo genético para executar enquanto as 92 soluções diferentes não forem encontradas. Avalie o custo computacional desta etapa.
