# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:41:25 2025

@author: Thales Alves

PLMC - Implementação da Metaheurística Algoritmo Genético

Estratégias adotadas
Tamanho das Populações: 1000
Estratégia de seleção dos pais: Torneio
Número de pais selecionados em cada população: 100
Número de filhos gerados por Crossover: 600
Número de filhos gerados por Mutação: 300
Estratégia adotada para o Crossover: Intercala os valores dos pais. Ex: [0, 1] + [2, 3] = [0, 2, 1, 3]
Estratégia adotada para a Mutação: Remove uma posicao, testa e preenche com valores aleatórios se necessário
Citério de parada: Testa se não houve melhora na aptidão nas últimas 50 execuções
"""

import random
import numpy as np

# -------------------- FUNÇÕES AUXILIARES --------------------
def calculaCobertura(solucao, numFazendas, distanciasFazendas, coberturaAntena):
    """
    Calcula quantas fazendas foram cobertas pela solução recebida via parâmetro
        
    parâmetros:
        solucao: int[] - vetor com as posicoes das antenas
        numFazendas: int - numero total de fazendas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int - quantidade de fazendas cobertas
        
    """
    
    fazendasCobertas = [0] * numFazendas
    
    for fazendaA in solucao:
        for fazendaB in range(numFazendas):
            if distanciasFazendas[fazendaA][fazendaB] <= coberturaAntena:
                fazendasCobertas[fazendaB] = 1
                
    return sum(fazendasCobertas)


def dividirVetor(vetor, quantidadeGrupos):
    """
    Divide o vetor em N vetores iguais (quanto for possível) e os retorna em uma matriz

    parâmetros:
        vetor: [] - vetor de quaisquer elementos
        quantidadeGrupos: int - quantidade de vetores a serem gerados
        
    saída:
        [][] - matriz resultante
    """
    
    return np.array_split(vetor, quantidadeGrupos)

def calculaAptidao(solucao, distanciasFazendas, numFazendas, coberturaAntena):
    """
    Calcula a aptidão da solução recebida
    
    parâmetros:
        solucao: int[] - vetor com as posicoes das antenas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        numFazendas: int - numero total de fazendas
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        float - razão entre a redundancia (cobertura) e a quantidade de antenas
    """
    
    redundancia = calculaCobertura(solucao, numFazendas, distanciasFazendas, coberturaAntena)
    """
    redundancia = 0
    
    for fazendaAntena in solucao:
        for posicaoFazenda in range(numFazendas):
            if distanciasFazendas[fazendaAntena][posicaoFazenda] <= coberturaAntena:
                redundancia += 1
    """
    
    return redundancia/len(solucao)

def selecionarPais(populacao, quantidadePais, distanciasFazendas, numFazendas, coberturaAntena):
    """
    Seleciona os melhores pais utilizando a estratégia torneio.
    
    parâmetros:
        populacao: int[][] - matriz onde cada 'linha' se refere a um indivíduo e cada 'coluna' se refere a uma posição da antena (a fazenda em questão)
        quantidadePais: int - quantidade de pais a ser selecionada
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        numFazendas: int - numero total de fazendas
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int[][] - matriz representando os pais selecionados no torneio
    """
    
    posicoes = np.random.choice(
        np.arange(len(populacao)),
        size=len(populacao),
        replace=False
    )
    
    gruposPosicoes = dividirVetor(posicoes, quantidadePais)
    melhores = []
    selecionados = []
    
    for posicoes in gruposPosicoes:
        melhorAptidao = 0
        posicaoMelhorRedundancia = None
        
        for posicaoIndividuo in posicoes:
            aptidaoAtual = calculaAptidao(populacao[posicaoIndividuo], distanciasFazendas, numFazendas, coberturaAntena)
            
            if aptidaoAtual > melhorAptidao:
                melhorAptidao = aptidaoAtual
                posicaoMelhorRedundancia = posicaoIndividuo
                
        melhores.append(posicaoMelhorRedundancia.item())
        
    for posicao in melhores:
        selecionados.append(populacao[posicao])
        
    return selecionados

def gerarFilho(pai1, pai2, numFazendas, distanciasFazendas, coberturaAntena):
    """
    Gera um filho por crossover a partir de dois pais distintos
    
    parâmetros:
        pai1: int[] - vetor contendo as posições das antenas
        pai2: int[] - vetor contendo as posições das antenas
        numFazendas: int - numero total de fazendas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int[] - filho gerado a partir dos dois pais recebidos
    """
    
    filhoFinal = set()
    cobreTodasFazendas = False
    
    maiorPai = max(len(pai1), len(pai2))
    
    for i in range(maiorPai):
        if i < len(pai1) and not pai1[i] in filhoFinal:
            filhoFinal.add(pai1[i])
            
            cobreTodasFazendas = calculaCobertura(list(filhoFinal), numFazendas, distanciasFazendas, coberturaAntena) == numFazendas
            
            if cobreTodasFazendas:
                break
            
        if i < len(pai2) and not pai2[i] in filhoFinal:
            filhoFinal.add(pai2[i])
            
            cobreTodasFazendas = calculaCobertura(list(filhoFinal), numFazendas, distanciasFazendas, coberturaAntena) == numFazendas
            
            if cobreTodasFazendas:
                break
        
    while not cobreTodasFazendas:
        novaAntena = random.randint(0, numFazendas)
        
        if not novaAntena in filhoFinal:
            filhoFinal.add(novaAntena)
            
            cobreTodasFazendas = calculaCobertura(list(filhoFinal), numFazendas, distanciasFazendas, coberturaAntena) == numFazendas
    
    return list(filhoFinal)

def gerarFilhosPorCrossover(pais, numFilhos, numFazendas, distanciasFazendas, coberturaAntena):
    """
    Dado o número de filhos a serem gerados, sorteia dois pais e gera um filho
    
    parâmetros:
        pais: int[][] - matriz contendo as posições das antenas
        numFilhos: int - número de filhos desejado
        numFazendas: int - numero total de fazendas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int[][] - matriz de filhos gerados
    """
    
    filhos = []
    
    for i in range(numFilhos):
        posicaoPai1, posicaoPai2 = np.random.choice(
            np.arange(len(pais)),
            size=2,
            replace=False
        )
        
        filhoGerado = gerarFilho(pais[posicaoPai1], pais[posicaoPai2], numFazendas, distanciasFazendas, coberturaAntena)
        
        filhos.append(filhoGerado)
        
    return filhos

def mutarPai(pai, numFazendas, distanciasFazendas, coberturaAntena):
    """
    Dado um pai, gera um filho por mutação
    
    parâmetros:
        pai: int[] - vetor contendo as posições das antenas
        numFazendas: int - numero total de fazendas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int[] - filhos gerados
    """
    
    filho = pai[:]
    
    antenaRemovida = random.choice(filho)
    filho.remove(antenaRemovida)
    
    while calculaCobertura(filho, numFazendas, distanciasFazendas, coberturaAntena) < numFazendas:
        antenasDisponiveis = set(range(numFazendas)) - set(filho)
        
        if antenasDisponiveis:
            novaAntena = random.choice(list(antenasDisponiveis))
            filho.append(novaAntena)
            
    return filho

def gerarFilhosPorMutacao(pais, numFilhos, numFazendas, distanciasFazendas, coberturaAntena):
    """
    Dado o número de filhos a serem gerados, sorteia um pai e realiza a mutação
    
    parâmetros:
        pais: int[][] - matriz contendo as posições das antenas
        numFilhos: int - número de filhos desejado
        numFazendas: int - numero total de fazendas
        distanciasFazendas: int[][] - matriz com a distância entre cada fazenda (km)
        coberturaAntena: int - a distancia coberta pela antena (km)
        
    saída:
        int[][] - filhos gerados
    """
    
    filhos = []
    
    for i in range(numFilhos):
        posicaoPai = random.randint(0, len(pais) - 1)
        
        filhoGerado = mutarPai(pais[posicaoPai], numFazendas, distanciasFazendas, coberturaAntena)
        
        filhos.append(filhoGerado)
        
    return filhos

# -------------------- CÓDIGO PRINCIPAL --------------------
distanciasFazendas = [
    [ 0.0,  3.6,  16.5,  28.4,  14.8,  29.1,  25.3,  25.6,  25.1,  22.5,  23.0,  14.6,  17.0,  19.2,  25.6,  25.8,  29.1,  37.5,  27.0,  33.1,  39.8,  41.9,  33.2,  36.2,  37.6,  35.7,  34.9,  36.8,  35.4,  36.3,  38.1,  42.0,  43.4,  43.1,  40.3,  42.4,  47.5,  46.5,  41.6,  50.5],
    [3.6,  0.0,  18.0,  30.1,  11.7,  30.4,  26.5,  26.7,  26.0,  22.8,  19.7,  12.5,  13.4,  16.0,  22.0,  24.8,  28.3,  36.9,  24.0,  31.6,  38.9,  41.0,  29.8,  32.7,  35.8,  33.6,  31.6,  33.2,  32.8,  33.7,  34.7,  38.5,  41.7,  41.1,  37.1,  38.9,  46.0,  44.9,  39.1,  49.2],
    [16.5,  18.0,  0.0,  12.0,  29.2,  12.6,  8.9,  9.4,  9.2,  8.5,  36.1,  15.6,  26.4,  24.2,  35.8,  17.0,  18.2,  25.1,  28.6,  25.0,  28.5,  30.1,  37.5,  42.4,  30.0,  30.4,  38.4,  42.4,  32.9,  33.8,  41.4,  48.1,  35.1,  36.0,  41.7,  46.9,  37.9,  37.5,  38.1,  39.8],
    [28.4,  30.1,  12.0,  0.0,  41.0,  3.0,  5.0,  5.7,  7.1,  11.4,  47.7,  25.6,  37.4,  34.0,  46.5,  20.6,  19.2,  22.1,  36.4,  26.8,  26.2,  27.0,  45.8,  51.3,  31.4,  33.6,  46.2,  51.0,  37.8,  38.6,  48.9,  56.6,  35.2,  37.0,  48.1,  54.7,  36.2,  36.5,  41.7,  37.0],
    [14.8,  11.7,  29.2,  41.0,  0.0,  41.0,  37.0,  37.1,  36.1,  32.4,  8.5,  18.4,  9.4,  15.6,  13.3,  31.3,  35.3,  43.8,  23.3,  36.4,  44.9,  47.2,  25.3,  26.1,  39.6,  36.1,  27.5,  27.0,  33.3,  34.1,  30.3,  31.4,  45.3,  43.9,  34.0,  33.0,  50.2,  48.8,  39.4,  53.9],
    [29.1,  30.4,  12.6,  3.0,  41.0,  0.0,  4.0,  4.1,  5.4,  9.8,  47.3,  24.7,  36.7,  32.9,  45.6,  18.4,  16.6,  19.1,  34.7,  24.2,  23.2,  24.0,  44.1,  49.7,  28.6,  31.1,  44.4,  49.4,  35.5,  36.2,  47.0,  54.9,  32.3,  34.2,  46.0,  52.8,  33.2,  33.5,  39.1,  34.0],
    [25.3,  26.5,  8.9,  5.0,  37.0,  4.0,  0.0,  1.0,  2.2,  6.4,  43.3,  20.9,  32.8,  29.2,  41.8,  15.8,  14.9,  19.1,  31.4,  22.5,  23.0,  24.2,  40.8,  46.3,  27.2,  29.1,  41.2,  46.0,  33.0,  33.8,  43.9,  51.6,  31.4,  33.0,  43.2,  49.7,  33.0,  33.1,  37.1,  34.2],
    [25.6,  26.7,  9.4,  5.7,  37.1,  4.1,  1.0,  0.0,  1.4,  5.8,  43.2,  20.6,  32.6,  28.8,  41.5,  15.0,  13.9,  18.1,  30.8,  21.5,  22.0,  23.2,  40.2,  45.8,  26.2,  28.2,  40.6,  45.5,  32.2,  33.0,  43.3,  51.0,  30.4,  32.0,  42.4,  49.0,  32.0,  32.1,  36.2,  33.2],
    [25.1,  26.0,  9.2,  7.1,  36.1,  5.4,  2.2,  1.4,  0.0,  4.5,  42.1,  19.4,  31.4,  27.5,  40.3,  13.6,  12.6,  17.3,  29.4,  20.2,  21.1,  22.4,  38.8,  44.4,  25.0,  26.8,  39.2,  44.1,  30.8,  31.6,  41.9,  49.6,  29.3,  30.8,  41.0,  47.6,  31.0,  31.0,  34.9,  32.4],
    [22.5,  22.8,  8.5,  11.4,  32.4,  9.8,  6.4,  5.8,  4.5,  0.0,  38.0,  15.1,  27.2,  23.1,  35.9,  9.8,  10.0,  16.6,  25.0,  17.3,  19.9,  21.5,  34.4,  40.0,  22.2,  23.4,  34.8,  39.7,  26.9,  27.8,  37.5,  45.2,  27.0,  28.2,  36.9,  43.3,  29.4,  29.2,  31.4,  31.3],
    [23.0,  19.7,  36.1,  47.7,  8.5,  47.3,  43.3,  43.2,  42.1,  38.0,  0.0,  23.0,  11.2,  17.1,  7.6,  34.9,  39.1,  47.1,  22.8,  38.5,  47.5,  49.8,  21.5,  20.4,  40.8,  36.6,  23.7,  21.6,  32.6,  33.2,  26.0,  25.0,  46.0,  44.2,  30.4,  27.5,  51.3,  49.6,  38.2,  55.2],
    [14.6,  12.5,  15.6,  25.6,  18.4,  24.7,  20.9,  20.6,  19.4,  15.1,  23.0,  0.0,  12.0,  8.6,  20.9,  13.0,  17.0,  25.6,  13.9,  19.2,  27.0,  29.2,  22.2,  26.9,  23.3,  21.2,  23.3,  26.9,  21.0,  22.0,  26.4,  32.6,  29.2,  28.6,  27.5,  31.6,  33.6,  32.4,  27.2,  36.9],
    [17.0,  13.4,  26.4,  37.4,  9.4,  36.7,  32.8,  32.6,  31.4,  27.2,  11.2,  12.0,  0.0,  6.4,  9.4,  23.8,  27.9,  36.1,  13.9,  27.8,  36.7,  38.9,  17.0,  19.3,  30.6,  26.9,  19.0,  19.9,  23.9,  24.7,  22.0,  25.1,  36.1,  34.7,  25.2,  25.7,  41.2,  39.7,  30.0,  45.0],
    [19.2,  16.0,  24.2,  34.0,  15.6,  32.9,  29.2,  28.8,  27.5,  23.1,  17.1,  8.6,  6.4,  0.0,  13.0,  18.1,  22.2,  30.1,  8.2,  21.5,  30.5,  32.7,  14.3,  18.4,  24.2,  20.5,  15.8,  18.6,  17.9,  18.8,  19.0,  24.2,  29.7,  28.3,  21.1,  23.7,  34.8,  33.3,  24.2,  38.6],
    [25.6,  22.0,  35.8,  46.5,  13.3,  45.6,  41.8,  41.5,  40.3,  35.9,  7.6,  20.9,  9.4,  13.0,  0.0,  31.0,  35.1,  42.6,  16.6,  33.2,  42.4,  44.6,  13.9,  13.0,  34.9,  30.4,  16.1,  14.1,  25.8,  26.4,  18.4,  18.1,  39.8,  37.7,  22.8,  20.1,  45.2,  43.4,  31.1,  49.2],
    [25.8,  24.8,  17.0,  20.6,  31.3,  18.4,  15.8,  15.0,  13.6,  9.8,  34.9,  13.0,  23.8,  18.1,  31.0,  0.0,  4.1,  12.5,  17.1,  8.1,  14.1,  16.3,  26.4,  32.3,  13.0,  13.6,  26.4,  31.8,  17.2,  18.0,  28.8,  37.1,  18.4,  19.0,  27.6,  34.7,  21.9,  21.2,  21.5,  24.7],
    [29.1,  28.3,  18.2,  19.2,  35.3,  16.6,  14.9,  13.9,  12.6,  10.0,  39.1,  17.0,  27.9,  22.2,  35.1,  4.1,  0.0,  8.6,  20.6,  7.6,  10.8,  12.8,  29.7,  35.7,  12.4,  14.4,  29.5,  35.1,  19.1,  19.8,  31.8,  40.3,  17.0,  18.2,  30.0,  37.6,  19.6,  19.2,  22.5,  21.9],
    [37.5,  36.9,  25.1,  22.1,  43.8,  19.1,  19.1,  18.1,  17.3,  16.6,  47.1,  25.6,  36.1,  30.1,  42.6,  12.5,  8.6,  0.0,  27.0,  10.2,  4.1,  5.1,  35.4,  41.4,  12.2,  16.6,  34.7,  40.6,  22.5,  22.8,  36.4,  45.4,  13.9,  16.4,  33.6,  42.1,  14.1,  14.6,  23.6,  15.1],
    [27.0,  24.0,  28.6,  36.4,  23.3,  34.7,  31.4,  30.8,  29.4,  25.0,  22.8,  13.9,  13.9,  8.2,  16.6,  17.1,  20.6,  27.0,  0.0,  17.1,  26.3,  28.4,  9.4,  15.2,  18.4,  13.9,  9.9,  14.8,  10.0,  10.8,  12.8,  20.2,  23.3,  21.4,  13.6,  18.4,  28.7,  26.9,  16.1,  32.6],
    [33.1,  31.6,  25.0,  26.8,  36.4,  24.2,  22.5,  21.5,  20.2,  17.3,  38.5,  19.2,  27.8,  21.5,  33.2,  8.1,  7.6,  10.2,  17.1,  0.0,  9.2,  11.4,  25.2,  31.3,  5.0,  7.1,  24.5,  30.4,  12.5,  13.0,  26.2,  35.2,  10.4,  11.0,  23.7,  32.0,  14.4,  13.4,  15.0,  17.7],
    [39.8,  38.9,  28.5,  26.2,  44.9,  23.2,  23.0,  22.0,  21.1,  19.9,  47.5,  27.0,  36.7,  30.5,  42.4,  14.1,  10.8,  4.1,  26.3,  9.2,  0.0,  2.2,  34.0,  40.0,  9.5,  14.3,  33.1,  39.1,  20.4,  20.6,  34.5,  43.6,  10.0,  12.7,  31.3,  40.0,  10.0,  10.4,  20.6,  11.4],
    [41.9,  41.0,  30.1,  27.0,  47.2,  24.0,  24.2,  23.2,  22.4,  21.5,  49.8,  29.2,  38.9,  32.7,  44.6,  16.3,  12.8,  5.1,  28.4,  11.4,  2.2,  0.0,  36.0,  42.0,  11.2,  16.1,  35.1,  41.0,  22.2,  22.4,  36.3,  45.4,  10.6,  13.6,  33.0,  41.8,  9.5,  10.3,  21.9,  10.0],
    [33.2,  29.8,  37.5,  45.8,  25.3,  44.1,  40.8,  40.2,  38.8,  34.4,  21.5,  22.2,  17.0,  14.3,  13.9,  26.4,  29.7,  35.4,  9.4,  25.2,  34.0,  36.0,  0.0,  6.1,  25.1,  20.1,  2.2,  5.4,  14.3,  14.6,  5.0,  10.8,  28.9,  26.2,  8.9,  9.4,  34.2,  32.3,  18.4,  38.3],
    [36.2,  32.7,  42.4,  51.3,  26.1,  49.7,  46.3,  45.8,  44.4,  40.0,  20.4,  26.9,  19.3,  18.4,  13.0,  32.3,  35.7,  41.4,  15.2,  31.3,  40.0,  42.0,  6.1,  0.0,  31.0,  26.0,  7.1,  1.4,  20.1,  20.2,  7.2,  5.8,  34.5,  31.8,  12.2,  7.1,  39.8,  37.9,  23.4,  43.9],
    [37.6,  35.8,  30.0,  31.4,  39.6,  28.6,  27.2,  26.2,  25.0,  22.2,  40.8,  23.3,  30.6,  24.2,  34.9,  13.0,  12.4,  12.2,  18.4,  5.0,  9.5,  11.2,  25.1,  31.0,  0.0,  5.0,  24.0,  30.0,  11.0,  11.2,  25.2,  34.2,  5.8,  6.0,  21.8,  30.6,  10.6,  9.2,  11.4,  14.4],
    [35.7,  33.6,  30.4,  33.6,  36.1,  31.1,  29.1,  28.2,  26.8,  23.4,  36.6,  21.2,  26.9,  20.5,  30.4,  13.6,  14.4,  16.6,  13.9,  7.1,  14.3,  16.1,  20.1,  26.0,  5.0,  0.0,  19.0,  25.0,  6.1,  6.3,  20.2,  29.3,  9.4,  7.8,  17.1,  25.7,  14.8,  13.0,  8.1,  18.8],
    [34.9,  31.6,  38.4,  46.2,  27.5,  44.4,  41.2,  40.6,  39.2,  34.8,  23.7,  23.3,  19.0,  15.8,  16.1,  26.4,  29.5,  34.7,  9.9,  24.5,  33.1,  35.1,  2.2,  7.1,  24.0,  19.0,  0.0,  6.0,  13.0,  13.2,  3.2,  10.8,  27.5,  24.7,  6.7,  8.5,  32.8,  30.8,  16.6,  36.9],
    [36.8,  33.2,  42.4,  51.0,  27.0,  49.4,  46.0,  45.5,  44.1,  39.7,  21.6,  26.9,  19.9,  18.6,  14.1,  31.8,  35.1,  40.6,  14.8,  30.4,  39.1,  41.0,  5.4,  1.4,  30.0,  25.0,  6.0,  0.0,  19.0,  19.1,  5.8,  5.7,  33.4,  30.6,  10.8,  6.0,  38.6,  36.7,  22.1,  42.8],
    [35.4,  32.8,  32.9,  37.8,  33.3,  35.5,  33.0,  32.2,  30.8,  26.9,  32.6,  21.0,  23.9,  17.9,  25.8,  17.2,  19.1,  22.5,  10.0,  12.5,  20.4,  22.2,  14.3,  20.1,  11.0,  6.1,  13.0,  19.0,  0.0,  1.0,  14.1,  23.2,  14.6,  12.1,  11.2,  19.6,  19.9,  18.0,  6.3,  24.0],
    [36.3,  33.7,  33.8,  38.6,  34.1,  36.2,  33.8,  33.0,  31.6,  27.8,  33.2,  22.0,  24.7,  18.8,  26.4,  18.0,  19.8,  22.8,  10.8,  13.0,  20.6,  22.4,  14.6,  20.2,  11.2,  6.3,  13.2,  19.1,  1.0,  0.0,  14.0,  23.1,  14.3,  11.7,  10.8,  19.4,  19.6,  17.7,  5.4,  23.8],
    [38.1,  34.7,  41.4,  48.9,  30.3,  47.0,  43.9,  43.3,  41.9,  37.5,  26.0,  26.4,  22.0,  19.0,  18.4,  28.8,  31.8,  36.4,  12.8,  26.2,  34.5,  36.3,  5.0,  7.2,  25.2,  20.2,  3.2,  5.8,  14.1,  14.0,  0.0,  9.1,  28.1,  25.2,  5.0,  5.8,  33.2,  31.3,  16.5,  37.3],
    [42.0,  38.5,  48.1,  56.6,  31.4,  54.9,  51.6,  51.0,  49.6,  45.2,  25.0,  32.6,  25.1,  24.2,  18.1,  37.1,  40.3,  45.4,  20.2,  35.2,  43.6,  45.4,  10.8,  5.8,  34.2,  29.3,  10.8,  5.7,  23.2,  23.1,  9.1,  0.0,  37.0,  34.1,  13.2,  4.5,  42.1,  40.1,  25.2,  46.2],
    [43.4,  41.7,  35.1,  35.2,  45.3,  32.3,  31.4,  30.4,  29.3,  27.0,  46.0,  29.2,  36.1,  29.7,  39.8,  18.4,  17.0,  13.9,  23.3,  10.4,  10.0,  10.6,  28.9,  34.5,  5.8,  9.4,  27.5,  33.4,  14.6,  14.3,  28.1,  37.0,  0.0,  3.2,  24.0,  33.0,  5.4,  3.6,  12.2,  9.5],
    [43.1,  41.1,  36.0,  37.0,  43.9,  34.2,  33.0,  32.0,  30.8,  28.2,  44.2,  28.6,  34.7,  28.3,  37.7,  19.0,  18.2,  16.4,  21.4,  11.0,  12.7,  13.6,  26.2,  31.8,  6.0,  7.8,  24.7,  30.6,  12.1,  11.7,  25.2,  34.1,  3.2,  0.0,  21.0,  30.0,  8.1,  6.1,  9.1,  12.2],
    [40.3,  37.1,  41.7,  48.1,  34.0,  46.0,  43.2,  42.4,  41.0,  36.9,  30.4,  27.5,  25.2,  21.1,  22.8,  27.6,  30.0,  33.6,  13.6,  23.7,  31.3,  33.0,  8.9,  12.2,  21.8,  17.1,  6.7,  10.8,  11.2,  10.8,  5.0,  13.2,  24.0,  21.0,  0.0,  9.0,  29.0,  27.0,  12.0,  33.1],
    [42.4,  38.9,  46.9,  54.7,  33.0,  52.8,  49.7,  49.0,  47.6,  43.3,  27.5,  31.6,  25.7,  23.7,  20.1,  34.7,  37.6,  42.1,  18.4,  32.0,  40.0,  41.8,  9.4,  7.1,  30.6,  25.7,  8.5,  6.0,  19.6,  19.4,  5.8,  4.5,  33.0,  30.0,  9.0,  0.0,  38.0,  36.0,  21.0,  42.0],
    [47.5,  46.0,  37.9,  36.2,  50.2,  33.2,  33.0,  32.0,  31.0,  29.4,  51.3,  33.6,  41.2,  34.8,  45.2,  21.9,  19.6,  14.1,  28.7,  14.4,  10.0,  9.5,  34.2,  39.8,  10.6,  14.8,  32.8,  38.6,  19.9,  19.6,  33.2,  42.1,  5.4,  8.1,  29.0,  38.0,  0.0,  2.0,  17.0,  4.1],
    [46.5,  44.9,  37.5,  36.5,  48.8,  33.5,  33.1,  32.1,  31.0,  29.2,  49.6,  32.4,  39.7,  33.3,  43.4,  21.2,  19.2,  14.6,  26.9,  13.4,  10.4,  10.3,  32.3,  37.9,  9.2,  13.0,  30.8,  36.7,  18.0,  17.7,  31.3,  40.1,  3.6,  6.1,  27.0,  36.0,  2.0,  0.0,  15.0,  6.1],
    [41.6,  39.1,  38.1,  41.7,  39.4,  39.1,  37.1,  36.2,  34.9,  31.4,  38.2,  27.2,  30.0,  24.2,  31.1,  21.5,  22.5,  23.6,  16.1,  15.0,  20.6,  21.9,  18.4,  23.4,  11.4,  8.1,  16.6,  22.1,  6.3,  5.4,  16.5,  25.2,  12.2,  9.1,  12.0,  21.0,  17.0,  15.0,  0.0,  21.0],
    [50.5,  49.2,  39.8,  37.0,  53.9,  34.0,  34.2,  33.2,  32.4,  31.3,  55.2,  36.9,  45.0,  38.6,  49.2,  24.7,  21.9,  15.1,  32.6,  17.7,  11.4,  10.0,  38.3,  43.9,  14.4,  18.8,  36.9,  42.8,  24.0,  23.8,  37.3,  46.2,  9.5,  12.2,  33.1,  42.0,  4.1,  6.1,  21.0,  0.0]
]

numFazendas = len(distanciasFazendas)
tamanhoPopulacao = 1000
coberturaAntena = 10

populacao = []
numPais = 100
numFilhosCrossover = 600
numFilhosMutacao = 300

melhorAptidao = 0
melhorSolucao = None

vezesManteveAptidao = 0
maximaVezesManteveAptidao = 50

# Gerando a população inicial
for _ in range(tamanhoPopulacao):
    solucao = []
    fazendasCobertas = 0
    
    while(fazendasCobertas < numFazendas):
        antenasDisponiveis = set(range(numFazendas)) - set(solucao)
        fazenda = random.choice(list(antenasDisponiveis))
                
        coberturaAtual = calculaCobertura(solucao + [fazenda], numFazendas, distanciasFazendas, coberturaAntena)
        
        if coberturaAtual > fazendasCobertas:
            solucao.append(fazenda)
            fazendasCobertas = coberturaAtual
            
    aptidaoAtual = calculaAptidao(solucao, distanciasFazendas, numFazendas, coberturaAntena)
    
    if aptidaoAtual > melhorAptidao:
        melhorAptidao = aptidaoAtual
        melhorSolucao = solucao[:]
            
    populacao.append(solucao)
            
# Gerando demais populações
while vezesManteveAptidao < maximaVezesManteveAptidao:
    vezesManteveAptidao += 1
    # Torneio
    paisSelecionados = selecionarPais(populacao, numPais, distanciasFazendas, numFazendas, coberturaAntena)
    
    filhosPorCrossover = gerarFilhosPorCrossover(paisSelecionados, numFilhosCrossover, numFazendas, distanciasFazendas, coberturaAntena)
    
    filhosPorMutacao = gerarFilhosPorMutacao(paisSelecionados, numFilhosMutacao, numFazendas, distanciasFazendas, coberturaAntena)
    
    populacao = paisSelecionados + filhosPorCrossover + filhosPorMutacao
    
    for solucao in populacao:
        aptidaoAtual = calculaAptidao(solucao, distanciasFazendas, numFazendas, coberturaAntena)
        
        if aptidaoAtual > melhorAptidao:
            melhorAptidao = aptidaoAtual
            melhorSolucao = solucao[:]
            vezesManteveAptidao = 0

print('Melhor aptidão: ', melhorAptidao)
print('Solução: ', melhorSolucao)
