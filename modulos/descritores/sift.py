# Este módulo possui funções relacionadas ao 
# método SIFT (Scale-Invariant Feature Transform) e 
# Bag of Visual Words (BoVW) 


import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time
from tqdm.notebook import tqdm  


def extrair_descritores_sift_locais(imagens, rotulos):
    """
    Extrai os descritores SIFT de um conjunto de imagens.

    Parâmetros:
    - imagens (list): Lista de imagens em formato NumPy.
    - rotulos (list): Lista de rotulos das imagens

    Retorno:
    - descritores_sift (list): Lista de descritores SIFT extraídas de cada imagem.
    """
    rotulos_validos = []
    # Redimensionamento das imagens para redução da complexidade
    # Existem imagens com mais de 2000x2000 no dataset 
    # O cálculo dos descritores pode ficar lento com esses tamanhos
    imagens = [cv2.resize(imagem, (imagem.shape[1] // 2, imagem.shape[0] // 2)) for imagem in imagens]
    
    # Calcula o total de imagens do conjunto
    total_imagens = len(imagens)
    # Lista para armazenar os descritores extraídos
    lista_descritores_sift = []
    # Cria o objeto sift 
    sift = cv2.SIFT_create()
    # Cria uma barra de progresso com o total de imagens
    with tqdm(total=total_imagens, desc="Extraindo descritores SIFT locais") as pbar:  
        for imagem, rotulo in zip(imagens, rotulos):
            if len(imagem.shape) > 2:
                # Se a imagem é colorida, converte para escala de cinza
                imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
            else:
                imagem_cinza = imagem  # Se a imagem já está em escala de cinza, mantém como está
            imagem_cinza = cv2.medianBlur(imagem_cinza, 3)
            # os  descritores SIFT são vetores de tamanho 128 que 
            # descrevem pontos de interesse (keypoints) de cada imagem
            # para cada imagem, extrai-se um número de keypoints e seus respectivos descritores 
            # descritores = uma matriz de tamanho num_keypoints x 128 valores numéricos
            ptos_int, descritores = sift.detectAndCompute(imagem_cinza, None)
            # checar se todos os descritores extraídos tem o 
            # tamanho correto (128)
            if descritores is not None and descritores.shape[1] == 128:
                lista_descritores_sift.append(descritores)
                # Adiciona o rótulo correspondente à lista de rótulos válidos
                rotulos_validos.append(rotulo)  
            pbar.update(1)  # Atualiza a barra de progresso
        # como cada imagem pode ter diferentes números de keypoints, o tipo objeto permite retornar um vetor de matrizes com tamanhos diferentes
        return np.array(lista_descritores_sift,dtype=object), np.array(rotulos_validos)


def treinar_kmeans(lista_descritores_sift):
    """
    Treina um modelo K-Means usando os descritores SIFT fornecidos.

    Parâmetros:
    - lista_descritores_sift (list of numpy arrays): Lista de arrays de descritores SIFT extraídos das imagens.
    
    Retorno:
    - kmeans (MiniBatchKMeans): O modelo treinado K-Means.
    - k (int): Número de grupos (clusters) usados para o treinamento.
    """
    k = 50 
    # n_init = número de vezes que o algoritmo KMeans será executado com diferentes 
    # centróides iniciais. 'auto' = valor será definido automaticamente
    kmeans = MiniBatchKMeans(n_clusters=k, n_init='auto', random_state=42)
    startTime = time.time()
    print('Treinando o modelo K-Means...')
    # o kmeans.fit requer uma matriz de tanho num_amostras x num_características
    # como a lista de desc é um vetor de matrizes de tamanho num_imagens x 1
    # e cada matriz dentro desse vetor é do tamanho num_keypoints x 128
    # np.vstack empilha verticalmente as matrizes resultando em uma única matriz 
    # de tamanho num_amostras x 128 (onde num_amostras = num_keypoints x num_imagens)

    # o KMeans agrupa as linhas (amostras) da lista de desc de todas as imagens
    # com base na proximidade (similaridade) de suas características (vetores de tamanho 128)
    # assim, vetores de características similares podem estar no mesmo grupo, pois descrevem 
    # keypoints similares ao longo de todas as imagens analisadas
    kmeans.fit(np.vstack(lista_descritores_sift))
    endTime = time.time()
    print(f'Tempo total de treinamento: {endTime - startTime:.2f} segundos')
    # retorna o modelo treinado kmeans e o número de clusters k
    return kmeans, k


def gerar_histogramas_bovw(lista_descritores_sift, modelo_kmeans, n_grupos):
    """
    Constrói histogramas globais para cada imagem usando a técnica Bag of Visual Words (BoVW).

    Parâmetros:
    - descritores_sift (list of numpy arrays): Lista de arrays contendo os descritores SIFT para cada imagem.
    - modelo_kmeans (MiniBatchKMeans): O modelo treinado de K-Means para predição de clusters.
    - n_grupos (int): Número de clusters (grupos) usados no modelo K-Means.

    Retorno:
    - histogramas (numpy array): Lista de histogramas, onde cada histograma é um histograma global para cada imagem representando a frequência de cada cluster.
    """
    histogramas = []
    # a lista de desc é um vetor de matrizes de tamanho num_imagens x 1
    # e cada matriz dentro desse vetor é do tamanho num_keypoints x 128
    for i in range(len(lista_descritores_sift)):
        # lista_descritores_sift[i]: uma matriz n_keypoints x 128 
        # cada linha da matriz acima contém um vetor de características de tamanho 128
        # kmeans.predict retorna um vetor (idx_arr) de tamanho n_keypoints x 1 
        # contendo o índice do grupo em que cada vetor de características pertence
        histogram = np.zeros(n_grupos)
        idx_arr = modelo_kmeans.predict(lista_descritores_sift[i])
        for d in range(len(idx_arr)):
            # cada elemento do vetor histogram corresponde a um cluster e 
            # o valor do elemento indica quantos descritores SIFT da imagem corrente 
            # foram atribuídos a esse cluster. Assim, o histogram armazena a 
            # distribuição dos descritores SIFT (cada vetor de tamanho 128) da imagem corrente 
            # em relação aos clusters (grupos) gerados pelo modelo KMeans 
            histogram[idx_arr[d]] += 1 
        histogramas.append(histogram)
    # retorna uma matriz de tamanho num_imagens x num_clusters (tamanho de cada histograma)
    # essa matriz será salva em arquivo como as features extraídas pelo método SIFT
    return np.array(histogramas,dtype=object)


def extrair_sift_treinamento(imagens, rotulos):
    """
    Extrai descritores SIFT, treina um modelo K-Means e gera histogramas BoVW para um conjunto de imagens.

    Parâmetros:
    - imagens (list of numpy arrays): Lista de imagens em formato NumPy para treinamento.

    Retorno:
    - lista_histogramas (numpy array): Lista de histogramas BoVW, onde cada histograma representa uma imagem.
    - modelo_kmeans (MiniBatchKMeans): O modelo K-Means treinado com os descritores SIFT.
    - n_grupos (int): Número de clusters (grupos) usados para o treinamento do modelo K-Means.
    """
    # Extrai os descritores SIFT locais para cada imagem
    lista_descritores_sift, rotulos = extrair_descritores_sift_locais(imagens, rotulos)
    
    # Treina o modelo K-Means usando os descritores SIFT extraídos
    modelo_kmeans, n_grupos = treinar_kmeans(lista_descritores_sift)
    
    # Gera histogramas BoVW para cada imagem com base no modelo K-Means treinado
    lista_histogramas = gerar_histogramas_bovw(lista_descritores_sift, modelo_kmeans, n_grupos)
    
    return lista_histogramas, modelo_kmeans, n_grupos, rotulos


def extrair_sift_teste(imagens, modelo_kmeans, n_grupos, rotulos):
    """
    Extrai descritores SIFT e gera histogramas BoVW para um conjunto de imagens de teste, usando um modelo K-Means pré-treinado.

    Parâmetros:
    - imagens (list of numpy arrays): Lista de imagens em formato NumPy para teste.
    - modelo_kmeans (MiniBatchKMeans): O modelo K-Means treinado usado para gerar os histogramas BoVW.
    - n_grupos (int): Número de clusters (grupos) usados no modelo K-Means para gerar os histogramas BoVW.

    Retorno:
    - lista_histogramas (numpy array): Lista de histogramas BoVW, onde cada histograma representa uma imagem.
    """
    # Extrai os descritores SIFT locais para cada imagem de teste
    lista_descritores_sift, rotulos = extrair_descritores_sift_locais(imagens, rotulos)
    
    # Gera histogramas BoVW para cada imagem de teste com base no modelo K-Means pré-treinado
    lista_histogramas = gerar_histogramas_bovw(lista_descritores_sift, modelo_kmeans, n_grupos)
    
    return lista_histogramas, rotulos
















