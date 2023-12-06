import random
import time
from math import sqrt
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt

# 1. Fa¸ca a defini¸c˜ao de quantos pontos devem ser gerados por regi˜ao. Escolha um valor 30 < Npontos < 60.
ponto_quadrante = 8  # Quantidade de pontos por quadrante
n_pontos = 4 * ponto_quadrante  # Quantidade de pontos 30 < Npontos < 60

pontos = [None] * n_pontos  # Array de pontos

# 2. Faça a definição da quantidade N de indivíduos em uma população e quantidade m´axima de gera¸c˜oes.
qntd_individuos = 5  # Quantidade de indivíduos
max_geracoes = 100000000  # Quantidade máxima de gerações
probabilidade_recombinacao = 0.95  # Probabilidade de recombina¸c˜ao (85% a 95%)
probabilidade_mutacao = 0.01  # Probabilidade de muta¸c˜ao (1%)


individuos = [None] * qntd_individuos  # Array de indivíduos

# quão espaçados os pontos gerados podem ser
rangelimit = 10

# contador de tempo
tempo = time.time()


# minha geração ANTES DE VER O ARQUIVO Q O PROFESSOR COLOCOU
# def gerar_ponto():
#     x = random.randint(0, rangelimit)
#     y = random.randint(0, rangelimit)
#     z = random.randint(0, rangelimit)
#     return [x, y, z]


# for i in range(n_pontos):
#     pontos[i] = gerar_ponto()


def generate_points(N): 
    x_partition = np.random.uniform(-10, 10, size=(N,3))
    y_partition = np.random.uniform(0, 20, size=(N,3))
    z_partition = np.random.uniform(-20, 0, size=(N,3))
    w_partition = np.random.uniform(0, 20, size=(N,3))

    x1 = np.array([[20,-20,-20]])
    x1 = np.tile(x1,(N,1))
    x_partition = x_partition+x1

    x1 = np.array([[-20,20,20]])
    x1 = np.tile(x1,(N,1))
    y_partition = y_partition+x1

    x1 = np.array([[-20,20,-20]])
    x1 = np.tile(x1,(N,1))
    z_partition = z_partition+x1

    x1 = np.array([[20,20,-20]])
    x1 = np.tile(x1,(N,1))
    w_partition = w_partition+x1   
    return np.concatenate((x_partition,y_partition,z_partition,w_partition), axis=0)

pontos = generate_points(ponto_quadrante)


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
            aptidao += distancia_entre_pontos(
                pontos[ponto_atual], pontos[self.cromossomo[i]]
            )
            ponto_atual = self.cromossomo[i]
        self.aptidao = aptidao
        return aptidao

    def gerar_cromossomo(self):
        cromosomo = []
        for i in range(n_pontos):
            cromosomo.append(i)
        random.shuffle(cromosomo)

        # garantir que o primeiro é sempre 0
        cromosomo.remove(0)
        cromosomo.insert(0, 0)

        return cromosomo

    def get_range(self, posicao_inicial, posicao_final):
        return self.cromossomo[posicao_inicial:posicao_final]

    def tentar_mutacao(self):
        if calc_probabilidade_mutacao():
            posicao = random.randint(1, n_pontos - 1)
            posicao2 = random.randint(1, n_pontos - 1)
            while posicao == posicao2:
                posicao2 = random.randint(1, n_pontos - 1)

            aux = self.cromossomo[posicao]
            self.cromossomo[posicao] = self.cromossomo[posicao2]
            self.cromossomo[posicao2] = aux
            self.aptidao = self.calcular_aptidao()

    def clone(self):
        return individuo(self.cromossomo)

    def toString(self):
        return str(self.cromossomo) + " - " + str(self.aptidao)


def distancia_entre_pontos(ponto1, ponto2):
    if len(ponto1) != len(ponto2):
        return False
    if len(ponto1) == 2:
        return sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2)
    return sqrt(
        (ponto1[0] - ponto2[0]) ** 2
        + (ponto1[1] - ponto2[1]) ** 2
        + (ponto1[2] - ponto2[2]) ** 2
    )


def roleta(aptidao, individuos, qntd_elitismo=0):
    if qntd_elitismo > 0:
        for i in range(qntd_elitismo):
            aptidao.remove(min(aptidao))

    soma = 0
    for i in range(len(aptidao)):
        aptidao[i] = aptidao[i] * -1
        soma += aptidao[i]

    for i in range(len(aptidao)):
        aptidao[i] = aptidao[i] / soma

    pais = [None] * 2
    posicoes = [None] * 2
    for i in range(2):
        valor = random.random()
        for j in range(len(aptidao)):
            valor -= aptidao[j]
            if valor <= 0:
                pais[i] = individuos[j]
                posicoes[i] = j
                break
    return [pais, posicoes]


def calc_probabilidade_mutacao():
    return random.random() <= probabilidade_mutacao

def calc_probabilidade_recombinacao():
    return random.random() <= probabilidade_recombinacao

def recombinacao_dois_pontos(pai1, pai2):
    if not calc_probabilidade_recombinacao():
        return False

    ponto1 = random.randint(1, n_pontos)
    ponto2 = random.randint(1, n_pontos)
    while ponto1 == ponto2:
        ponto2 = random.randint(0, n_pontos)

    if ponto1 > ponto2:
        aux = ponto1
        ponto1 = ponto2
        ponto2 = aux
    
    #aux2 = direita do ponto2 do pai2 + esquerda do ponto1 do pai2 + meio do pai 2
    aux2 = pai2[ponto2:n_pontos] + pai2[0:ponto1] + pai2[ponto1:ponto2]
    
    #aux1 = direita do ponto2 do pai1 + esquerda do ponto1 do pai1 + meio do pai 1
    aux1 = pai1[ponto2:n_pontos] + pai1[0:ponto1] + pai1[ponto1:ponto2]
    
    filho1 = [None] * n_pontos
    filho2 = [None] * n_pontos
    
    # coloca os meios
    for i in range(ponto1, ponto2):
        filho1[i] = pai2[i]
        filho2[i] = pai1[i]
 
    
    posAux1 = 0
    posAux2 = 0
    # coloca o final
    for i in range(ponto2, n_pontos):
        while aux1[posAux1] in filho1:
            posAux1 += 1
        filho1[i] = aux1[posAux1]
        while aux2[posAux2] in filho2:
            posAux2 += 1
        filho2[i] = aux2[posAux2]
        
    # coloca o inicio
    for i in range(0, ponto1):
        while aux1[posAux1] in filho1:
            posAux1 += 1
        filho1[i] = aux1[posAux1]
        while aux2[posAux2] in filho2:
            posAux2 += 1
        filho2[i] = aux2[posAux2]

    # garante que o primeiro é sempre 0
    filho1.remove(0)
    filho1.insert(0, 0)
    filho2.remove(0)
    filho2.insert(0, 0)

    #retorna cromossomo
    return [filho1, filho2]

for i in range(qntd_individuos):
    individuos[i] = individuo(None)


def logar(geracao_atual, melhor_individuo, geracao_do_melhor):
    print("---------------------------------------------------")
    print("Geração: ", geracao_atual)
    print("Melhor indivíduo: ", melhor_individuo.toString())
    print("Geração do melhor indivíduo: ", geracao_do_melhor)
    print("---------------------------------------------------")


# retorna a geracao do melhor individuo
def rodada(debug, elitismo=False, qntd_elitismo=0):
    melhor_individuo = individuos[0]
    geracao_do_melhor = 0

    geracao_atual = 0
    while geracao_atual < max_geracoes:
        aptidao = [None] * qntd_individuos

        individuos.sort(key=lambda x: x.aptidao)
        
        for i in range(qntd_individuos):
            aptidao[i] = individuos[i].aptidao
            if individuos[i].aptidao < melhor_individuo.aptidao:
                melhor_individuo = individuo(individuos[i].cromossomo)
                geracao_do_melhor = geracao_atual


        pais = [None] * 2
        posicoes = [None] * 2

        # 3. Projete o operador de seleção, baseado no método do torneio.
        [pais, posicoes] = roleta(aptidao, individuos, qntd_elitismo)

        # 4. Na etapa de recombinação, como este trata-se de um problema de combinatória, não pode haver pontos repetidos na sequência cromossômica. Desta maneira, pede-se que desenvolva uma variação do operador de recombinação de dois pontos. Assim, cada seção selecionada de modo aleatório deve ser propagada nos filhos e em seguida, a sequência genética do filho deve ser completada com os demais pontos sem repetição.
        # recombinou = recombinacaoUmPonto(pais[0], pais[1]);
        recombinou = recombinacao_dois_pontos(pais[0].cromossomo, pais[1].cromossomo)

        if recombinou:
            individuos[posicoes[0]] = individuo(recombinou[0])
            individuos[posicoes[1]] = individuo(recombinou[1])

        # 5. Na prole gerada, deve-se aplicar a mutação com probabilidade de 1%. Para o problema do caixeiro viajante, deve-se aplicar uma mutação que faz a troca de um gene por outro de uma mesma sequência cromossômica.
        for i in range(qntd_individuos):
            if not elitismo:
                individuos[i].tentar_mutacao()
            else:
                if i >= qntd_elitismo:
                    individuos[i].tentar_mutacao()

        # 6. O algoritmo deve parar quando atingir o máximo número de gerações ou quando a função custo atingir seu valor ótimo aceitável (de acordo com a regra descrita no slide 31/61).
        if geracao_atual - geracao_do_melhor > 5000000:
            if debug:
                logar(geracao_atual, melhor_individuo, geracao_do_melhor)
            break

        if geracao_atual % 10000 == 0:
            if debug:
                logar(geracao_atual, melhor_individuo, geracao_do_melhor)
        geracao_atual += 1

    if debug:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pontos[:,0], pontos[:,1], pontos[:,2], c='#248DD2', marker='o')

        # para cada ponto, desenha uma linha até o próximo
        for i in range(n_pontos - 1):
            p1 = pontos[melhor_individuo.cromossomo[i]]
            p2 = pontos[melhor_individuo.cromossomo[i + 1]]
            line, = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color='k')

        # desenha uma linha do ultimo ponto até o primeiro
        p1 = pontos[melhor_individuo.cromossomo[n_pontos - 1]]
        p2 = pontos[melhor_individuo.cromossomo[0]]
        line, = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color='k')

        plt.tight_layout()
        plt.show()
    return geracao_do_melhor

print(rodada(True, True, 1))


# rodada simples
# rodada(True);

# 7. Faça uma análise se de qual é a moda de gerações para obter uma solução aceitável. Além disso, analise se é necessário incluir um operador de elitismo em um grupo Ne de indivíduos.
# analise:
# apos rodar muitas vezes, percebi que com elitismo de 1 individuo, o algoritmo encontra a solução otima em menos gerações
# qntd_rodadas = 1000
# resultados_normais = [None] * qntd_rodadas
# resultados_elitista = [None] * qntd_rodadas
# for i in range(qntd_rodadas):
#     individuos_desse_teste = [None] * qntd_individuos
#     for j in range(qntd_individuos):
#         individuos_desse_teste[j] = individuo(None)
#         individuos[j] = individuos_desse_teste[j].clone()
#     resultados_normais[i] = rodada(False, False, 0)
#     for j in range(qntd_individuos):
#         individuos[j] = individuos_desse_teste[j].clone()
#     resultados_elitista[i] = rodada(False, True, 2)
#     if i % 100 == 0:
#         print("Rodada: ", i, " de ", qntd_rodadas)

# print("resultados normais: ", resultados_normais)
# print("media normal: ", sum(resultados_normais) / len(resultados_normais))
# print("resultados elitista: ", resultados_elitista)
# print("media elitista: ", sum(resultados_elitista) / len(resultados_elitista))
