import random
import time
from math import sqrt, ceil
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt


# contador de tempo
tempo = time.time()


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
                ponto_atual, self.cromossomo[i]
            )
            ponto_atual = self.cromossomo[i]
        self.aptidao = aptidao
        return aptidao

    def gerar_cromossomo(self):
        cromosomo = []
        for i in range(n_pontos):
            cromosomo.append(i)
        random.shuffle(cromosomo)

        return cromosomo

    def get_range(self, posicao_inicial, posicao_final):
        return self.cromossomo[posicao_inicial:posicao_final]

    def tentar_mutacao(self):
        if calc_probabilidade_mutacao():
            posicao = random.randint(0, n_pontos - 1)
            posicao2 = random.randint(0, n_pontos - 1)
            while posicao == posicao2:
                posicao2 = random.randint(0, n_pontos - 1)

            aux = self.cromossomo[posicao]
            self.cromossomo[posicao] = self.cromossomo[posicao2]
            self.cromossomo[posicao2] = aux
            self.aptidao = self.calcular_aptidao()

    def clone(self):
        return individuo(self.cromossomo)

    def toString(self):
        return str(self.cromossomo) + " - " + str(self.aptidao)
    
class populacao:
    def __init__(self, individuos):
        self.individuos = individuos

    def inicializar(self, tamanho):
        for i in range(tamanho):
            self.add_individuo(individuo(None))


    def inicializar(self, tamanho):
        for i in range(tamanho):
            self.add_individuo(individuo(None))

    def add_individuo(self, individuo):
        self.individuos.append(individuo)

    def get_individuo(self, posicao):
        return self.individuos[posicao]

    def retirar_pior(self):
        if(self.length() == 0):
            return None
        
        if(self.length() == 1):
            return self.individuos[0]

        pior = self.individuos[0]
        for i in range(1, self.length()):
            if self.individuos[i].aptidao > pior.aptidao:
                pior = self.individuos[i]
        self.individuos.remove(pior)
        return pior

    def get_melhor(self):
        if(self.length() == 0):
            return None
        
        if(self.length() == 1):
            return self.individuos[0]

        melhor = self.individuos[0]
        for i in range(1, self.length()):
            if self.individuos[i].aptidao < melhor.aptidao:
                melhor = self.individuos[i]
        return melhor
    
    def tentar_mutacao(self):
        for i in range(self.length()):
            self.individuos[i].tentar_mutacao()


    def torneio(self):
        aleatorio1 = random.randint(0, self.length() - 1)
        aleatorio2 = random.randint(0, self.length() - 1)

        while aleatorio1 == aleatorio2:
            aleatorio2 = random.randint(0, self.length() - 1)
        
        if self.individuos[aleatorio1].aptidao < self.individuos[aleatorio2].aptidao:
            return self.individuos[aleatorio1]
        else:
            return self.individuos[aleatorio2]
        
    def length(self):
        return len(self.individuos)


def distancia_entre_pontos(ponto1, ponto2):
    ponto1 = pontos[ponto1]
    ponto2 = pontos[ponto2]
    return sqrt(
        (ponto1[0] - ponto2[0]) ** 2
        + (ponto1[1] - ponto2[1]) ** 2
        + (ponto1[2] - ponto2[2]) ** 2
    )

def calc_probabilidade_mutacao():
    return random.random() <= probabilidade_mutacao

def calc_probabilidade_recombinacao():
    return random.random() <= probabilidade_recombinacao

def recombinacao_dois_pontos(pai1, pai2):
    if not calc_probabilidade_recombinacao():
        return [pai1, pai2]

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
    #retorna cromossomo
    return [filho1, filho2]


def mostrar_grafico(melhor_individuo):
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




# 1. Fa¸ca a defini¸c˜ao de quantos pontos devem ser gerados por regi˜ao. Escolha um valor 30 < Npontos < 60.
ponto_quadrante = 15  # Quantidade de pontos por quadrante
n_pontos = 4 * ponto_quadrante  # Quantidade de pontos 30 < Npontos < 60

pontos = [None] * n_pontos  # Array de pontos

# 2. Faça a definição da quantidade N de indivíduos em uma população e quantidade m´axima de gera¸c˜oes.
qntd_individuos = 100  # Quantidade de indivíduos em uma população (somente números pares pelo amor de deus)
max_geracoes = 1000000  # Quantidade máxima de gerações
probabilidade_recombinacao = 0.95  # Probabilidade de recombina¸c˜ao (85% a 95%)
probabilidade_mutacao = 0.01  # Probabilidade de muta¸c˜ao (1%)

pontos = generate_points(ponto_quadrante)


# retorna a geracao do melhor individuo
def rodada(qntd_elitismo=0):
    individuos = populacao([])
    individuos.inicializar(qntd_individuos)

    melhor_individuo = individuos.get_melhor()
    geracao_do_melhor = 0

    geracao_atual = 0
    while geracao_atual < max_geracoes:

        
        melhor_atual = individuos.get_melhor()
        novos_individuos = populacao([])
        
        if melhor_atual.aptidao < melhor_individuo.aptidao:
            print("melhor: ", melhor_atual.aptidao, " - ", melhor_atual.cromossomo, " - geracao: ", geracao_atual)
            melhor_individuo = melhor_atual
            geracao_do_melhor = geracao_atual
        
        if qntd_elitismo > 0:
            for i in range(qntd_elitismo):
                novos_individuos.add_individuo(individuos.get_melhor().clone())

        qnt_pares = individuos.length() // 2
        for _i in range(qnt_pares):
            # 3. Projete o operador de seleção, baseado no método do torneio.
            pai1 = individuos.torneio()
            pai2 = individuos.torneio()

            # 4. Na etapa de recombinação, como este trata-se de um problema de combinatória, não pode haver pontos repetidos na sequência cromossômica. Desta maneira, pede-se que desenvolva uma variação do operador de recombinação de dois pontos. Assim, cada seção selecionada de modo aleatório deve ser propagada nos filhos e em seguida, a sequência genética do filho deve ser completada com os demais pontos sem repetição.
            # recombinou = recombinacaoUmPonto(pais[0], pais[1]);
            recombinou = recombinacao_dois_pontos(pai1.cromossomo, pai2.cromossomo)

            if recombinou:
                novos_individuos.add_individuo(individuo(recombinou[0]))
                novos_individuos.add_individuo(individuo(recombinou[1]))
        
        if(qntd_elitismo > 0):
            for i in range(qntd_elitismo):
                novos_individuos.retirar_pior()
        
        # 5. Na prole gerada, deve-se aplicar a mutação com probabilidade de 1%. Para o problema do caixeiro viajante, deve-se aplicar uma mutação que faz a troca de um gene por outro de uma mesma sequência cromossômica.
        individuos = novos_individuos
        novos_individuos.tentar_mutacao()


        # 6. O algoritmo deve parar quando atingir o máximo número de gerações ou quando a função custo atingir seu valor ótimo aceitável (de acordo com a regra descrita no slide 31/61).
        if geracao_atual - geracao_do_melhor > 5000:
            break

        # if geracao_atual % 1000 == 0:
        #     print("geração: ", geracao_atual, " - melhor: ", melhor_individuo.aptidao, " - melhor: ", melhor_individuo.cromossomo)
        geracao_atual += 1

    mostrar_grafico(melhor_individuo)
    return geracao_do_melhor

resultado = rodada(1)
print(resultado)



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


                    



# rodada simples
# rodada(True);

# 7. Faça uma análise se de qual é a moda de gerações para obter uma solução aceitável. Além disso, analise se é necessário incluir um operador de elitismo em um grupo Ne de indivíduos.
# analise