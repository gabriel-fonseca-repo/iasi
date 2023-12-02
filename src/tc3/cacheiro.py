import random
import time
from math import ceil, sqrt

# 1. Fa¸ca a defini¸c˜ao de quantos pontos devem ser gerados por regi˜ao. Escolha um valor 30 < Npontos < 60.
Npontos = 50  # Quantidade de pontos 30 < Npontos < 60

# 2. Faça a definição da quantidade N de indivíduos em uma população e quantidade m´axima de gera¸c˜oes.
qntdIndividuos = 10  # Quantidade de indivíduos
maxGeracoes = 1  # Quantidade máxima de gerações
probabilidadeRecombinacao = 0.95  # Probabilidade de recombina¸c˜ao (85% a 95%)
probabilidadeMutacao = 0.01  # Probabilidade de muta¸c˜ao (1%)

individuos = [None] * qntdIndividuos  # Array de indivíduos

# contador de tempo
tempo = time.time()

class individuo:
  def __init__(self, cromossomo):
    if(cromossomo == None):
      self.cromossomo = self.gerarCromossomo()
    else:
      self.cromossomo = cromossomo
    self.calcularAptidao()
  
  def calcularAptidao(self):
    aptidao = 0
    ponto_atual = self.cromossomo[0]
    for i in range(Npontos):
      aptidao += distanciaEntrePontos(ponto_atual, self.cromossomo[i])
      ponto_atual = self.cromossomo[i]
    self.aptidao = aptidao
    return aptidao

  def gerarCromossomo(self):
    cromosomo = []
    for i in range(Npontos):
      cromosomo.append(gerarPonto())
    return cromosomo
  
  def getRange(self, posicaoInicial, posicaoFinal):
    return self.cromossomo[posicaoInicial:posicaoFinal]
  
  def tentarMutacao(self):
    if (probabilidadeMutacao()):
      posicao = random.randint(0, 7)
      self.cromossomo[posicao] = gerarGene()
      self.aptidao = self.calcularAptidao()

  def toString(self):
    return str(self.cromossomo) + " - " + str(self.aptidao)
  
def gerarPonto():
  x = random.randint(0, 30)
  y = random.randint(0, 30)
  z = random.randint(0, 30)
  return [x, y, z]

def distanciaEntrePontos(ponto1, ponto2):
  return sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2 + (ponto1[2] - ponto2[2]) ** 2)

def roleta(aptidao, individuos):
  # invertendo aptidao para que o maior valor seja o menor e vice-versa
  soma = 0
  for i in range(len(aptidao)):
    aptidao[i] = aptidao[i] * -1
    soma += aptidao[i]

  for i in range(len(aptidao)):
    aptidao[i] = aptidao[i]  / soma

  pais = [None] * 2
  for i in range(2):
    valor = random.random()
    for j in range(len(aptidao)):
      valor -= aptidao[j]
      if (valor <= 0):
        pais[i] = individuos[j]
        break
  return pais



for i in range(qntdIndividuos):
  individuos[i] = individuo(None)

geracaoAtual = 0
while (geracaoAtual < maxGeracoes):

  aptidao = [None] * qntdIndividuos

  for i in range(qntdIndividuos):
    aptidao[i] = individuos[i].aptidao
  
  pais = [None] * 2


  # 3. Projete o operador de seleção, baseado no método do torneio.
  pais = roleta(aptidao, individuos)

  
  # 4. Na etapa de recombinação, como este trata-se de um problema de combinatória, não pode haver pontos repetidos na sequência cromossômica. Desta maneira, pede-se que desenvolva uma variação do operador de recombina¸c˜ao de dois pontos. Assim, cada seção selecionada de modo aleatório deve ser propagada nos filhos e em seguida, a sequência genética do filho deve ser completada com os demais pontos sem repetição.





  print("---------------------------------------------------")
  print("Geração: ", geracaoAtual)
  print("---------------------------------------------------")
  geracaoAtual += 1

# 5. Na prole gerada, deve-se aplicar a muta¸c˜ao com probabilidade de 1%. Para o problema do caixeiro viajante, deve-se aplicar uma muta¸c˜ao que faz a troca de um gene por outro de uma mesma sequˆencia cromossˆomica.
# 6. O algoritmo deve parar quando atingir o m´aximo n´umero de gera¸c˜oes ou quando a fun¸c˜ao custo atingir seu valor ´otimo aceit´avel (de acordo com a regra descrita no slide 31/61).
# 7. Fa¸ca uma an´alise se de qual ´e a moda de gera¸c˜oes para obter uma solu¸c˜ao aceit´avel. Al´em disso, analise se ´e necess´ario incluir um operador de elitismo em um grupo Ne de indiv´ıduos.
