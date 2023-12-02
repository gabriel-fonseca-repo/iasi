import random
import time
from math import ceil

# 1. Faça a definição da quantidade N de indivíduos em uma população e quantidade máxima de geraçõoes.
N = 100  # Quantidade de indivíduos
max_geracoes = 100000000  # Quantidade máxima de gerações

probabilidade_recombinação = 0.95  # Probabilidade de recombinação (85% a 95%)
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
    if(cromossomo == None):
      self.cromossomo = self.gerarCromossomo()
    else:
      self.cromossomo = cromossomo
    self.aptidao = self.calcularAptidao()
  
  def calcularAptidao(self):
    aptidao = 0
    for i in range(8):
      for j in range(i + 1, 8):
        if (self.cromossomo[i] == self.cromossomo[j]):
          aptidao += 1
        elif (self.cromossomo[i] == self.cromossomo[j] - (j - i)):
          aptidao += 1
        elif (self.cromossomo[i] == self.cromossomo[j] + (j - i)):
          aptidao += 1
      
    return 28 - aptidao

  def gerarCromossomo(self):
    cromosomo = []
    for i in range(8):
      cromosomo.append(gerarGene())
    return cromosomo
  
  def getRange(self, posicaoInicial, posicaoFinal):
    return self.cromossomo[posicaoInicial:posicaoFinal]
  
  def tentarMutacao(self):
    if (probabilidadeMutacao()):
      posicao = random.randint(0, 7)
      self.cromossomo[posicao] = gerarGene()
      self.aptidao = self.calcularAptidao()

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
      if (r <= 0):
        pais[j] = individuos[k]
        break

  return pais
  
def recombinacaoUmPonto(pai, pai2):
  if(not probabilidadeRecombinacao()):
    return False
  posicao = random.randint(1, 7)

  filhos = [None] * 2

  filhos[0] = individuo(pai.getRange(0, posicao) + pai2.getRange(posicao, 8))
  filhos[1] = individuo(pai2.getRange(0, posicao) + pai.getRange(posicao, 8))

  return filhos;

def recombinacaoDoisPontos(pai, pai2):
  if(not probabilidadeRecombinacao()):
    return False
  posicao1 = random.randint(1, 7)
  posicao2 = random.randint(1, 7)

  if (posicao1 > posicao2):
    posicao1, posicao2 = posicao2, posicao1

  filhos = [None] * 2

  filhos[0] = individuo(pai.getRange(0, posicao1) + pai2.getRange(posicao1, posicao2) + pai.getRange(posicao2, 8))
  filhos[1] = individuo(pai2.getRange(0, posicao1) + pai.getRange(posicao1, posicao2) + pai2.getRange(posicao2, 8))

  return filhos;

def probabilidadeRecombinacao():
  rand = random.random()
  if (rand <= probabilidade_recombinação):
    return True
  else:
    return False

def probabilidadeMutacao():
  rand = random.random()
  if (rand <= probabilidade_mutacao):
    return True
  else:
    return False
  
# gerar um gene aleatório (0 a 7)
def gerarGene():
  return random.randint(0, 7)

for i in range(N):
  individuos[i] = individuo(None)
  
geracao_atual = 0
while(geracao_atual < max_geracoes):
  aptidao = [None] * N

  terminou = False

  for i in range(N):
    aptidao[i] = individuos[i].aptidao
    if (aptidao[i] == 28):
      if(not encontrar_todas_solucoes):
        print("Solução encontrada!")
        individuos[i].print()
        terminou = True
      else:
        if (individuos[i].toString() not in todas_solucoes):
          solucoes_encontradas += 1
          todas_solucoes[solucoes_encontradas - 1] = individuos[i].toString()
          if (solucoes_encontradas == total_solucoes):
            print("Todas as Soluções encontradas!")
            terminou = True
            break

  # 5. O algoritmo deve parar quando atingir o máximo número de gerações ou quando a função custo atingir seu valor ótimo.
  if (terminou):
    break
    



  # 2. Projete o operador de seleção, baseado no método da roleta.
  pais = roleta(aptidao, individuos);

  # 3. Na etapa de recombinação, escolha um entre os seguintes operadores: Recombinação de um ponto ou Recombinação de dois pontos. A probabilidade de recombinação nesta etapa deve ser entre 85 < pc < 95%.
  # fiz as duas opções para testar
  # recombinou = recombinacaoUmPonto(pais[0], pais[1]);
  recombinou = recombinacaoDoisPontos(pais[0], pais[1]);

  if (recombinou):
    # 4. Na prole gerada, deve-se aplicar a mutação com probabilidade de 1% (neste caso, é interessante avaliar os diferentes procedimentos exibidos).
    recombinou[0].tentarMutacao()
    recombinou[1].tentarMutacao()

    for i in range(N):
      if(individuos[i] == pais[0]):
        individuos[i] = recombinou[0]
      elif(individuos[i] == pais[1]):
        individuos[i] = recombinou[1]

  

  geracao_atual += 1
  if(geracao_atual % 100000 == 0):
    print("---------------------------------------------------")
    print("Geracao: ", geracao_atual)
    print("aptidao maxima: ", max(aptidao))
    if(encontrar_todas_solucoes):
      print("Soluções encontradas: ", solucoes_encontradas)
      if(geracao_atual == max_geracoes):
        if(solucoes_encontradas == 0):
          print("Nenhuma solução encontrada")
    print("---------------------------------------------------")


# tempo em horas, minutos e segundos
tempo_normal = time.time() - tempo
tempo_horas = ceil(tempo_normal // 3600)
tempo_normal %= 3600
tempo_minutos = ceil(tempo_normal // 60)
tempo_normal %= 60
tempo_segundos = ceil(tempo_normal)

print("Tempo de execução: ", tempo_horas, "h ", tempo_minutos, "m ", tempo_segundos, "s")

# De posse do primeiro resultado, aplique o seu projeto de algoritmo genético para executar enquanto as 92 soluções diferentes não forem encontradas. Avalie o custo computacional desta etapa.

  

