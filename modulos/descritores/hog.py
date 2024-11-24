import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

def extrair_hog(imagens, orientacoes=6, pixels_por_célula=(12, 12), células_por_bloco=(3, 3), visualizacao=False):
    """
    Extrai características HOG (Histogram of Oriented Gradients) de uma lista de imagens.

    :param imagens: Lista de imagens em formato de array numpy.
    :param orientacoes: Número de orientações do gradiente no histograma HOG. Default é 9.
    :param pixels_por_célula: Número de pixels por célula no cálculo do HOG. Default é (8, 8).
    :param células_por_bloco: Número de células por bloco no cálculo do HOG. Default é (2, 2).
    :param visualizacao: Se True, retorna a imagem HOG visualizada junto com as características. Default é False.

    :return: Array numpy contendo as características HOG de cada imagem.
    """
    # Se as imagens têm diferentes tamanhos, o vetor HOG resultante 
    # pode variar em comprimento. O redimensionamento das imagens pode
    # garantir que o vetor HOG tenha o mesmo tamanho para todas as imagens.
    tamanho_fixo=(224, 224)
    imagens = [cv2.resize(imagem, tamanho_fixo) for imagem in imagens]
    # Calcula o total de imagens do conjunto
    total_imagens = len(imagens)
    # Inicializa uma lista para armazenar as características HOG de cada imagem
    lista_hog = []
    with tqdm(total=total_imagens, desc="Extraindo características HOG") as pbar:
        # Itera sobre cada imagem na lista de imagens
        for imagem in imagens:
            # Verifica se a imagem é colorida (tem mais de 2 dimensões)
            if len(imagem.shape) > 2:
                # Converte a imagem de RGB para escala de cinza
                imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
            
            # Calcula as características HOG da imagem
            hog_features = hog(imagem, 
                               orientations=orientacoes, 
                               pixels_per_cell=pixels_por_célula, 
                               cells_per_block=células_por_bloco, 
                               visualize=visualizacao, 
                               feature_vector=True)
            
            # Adiciona as características HOG à lista
            lista_hog.append(hog_features)
            pbar.update(1)  # Atualiza a barra de progresso
    try:

        # Converte a lista de histogramas em um array numpy
        lista_hog_array = np.array(lista_hog)
        
        # REDUÇÃO DA DIMENSIONALIDADE COM PCA
        # n_components_pca: Número de componentes principais que serão mantidos
        n_samples, n_features = lista_hog_array.shape
        n_components_pca = 50
        
        pca = PCA(n_components=n_components_pca)
        lista_hog_reduzidos = pca.fit_transform(lista_hog_array)
        
        return lista_hog_reduzidos
    except ValueError:
        print("Erro: As características HOG têm tamanhos inconsistentes. Retornando lista.")
        return np.array(lista_hog)


def aplicar_pca(caracteristicas, n_componentes=50):
    """
    Aplica PCA para reduzir a dimensionalidade das características.

    :param caracteristicas: Array numpy contendo as características a serem reduzidas.
    :param n_componentes: Número de componentes principais a serem mantidos.

    :return: Array numpy com as características reduzidas.
    """
    pca = PCA(n_components=n_componentes)
    caracteristicas_reduzidas = pca.fit_transform(caracteristicas)
    return caracteristicas_reduzidas




