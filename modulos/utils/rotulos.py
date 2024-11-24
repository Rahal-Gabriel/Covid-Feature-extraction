# Este módulo possui funções utilitárias para
# codificação e decodificação de rótulos

import numpy as np
from sklearn import preprocessing

def codificar_rotulos_label(rotulos):
    """
    Codifica os rótulos de categorias usando LabelEncoder.

    Parâmetros:
    - rotulos (list of str): Lista de rótulos de categorias.

    Retorno:
    - rotulos_codificados (numpy array): Array de rótulos codificados como inteiros.
    - encoder (LabelEncoder): O objeto LabelEncoder usado para a codificação.
    """
    # Cria uma instância do LabelEncoder
    encoder = preprocessing.LabelEncoder()

    # Ajusta o encoder aos rótulos e os transforma em inteiros
    rotulos_codificados = encoder.fit_transform(rotulos)
    print('Rótulos codificados com LabelEncoder')
    return rotulos_codificados, encoder

def codificar_rotulos_onehot(rotulos):
    """
    Codifica os rótulos de categorias usando OneHotEncoding.

    Parâmetros:
    - rotulos (list of str): Lista de rótulos de categorias.

    Retorno:
    - rotulos_onehot (numpy array): Array 2D de rótulos codificados como vetores OneHot.
    - encoder (OneHotEncoder): O objeto OneHotEncoder usado para a codificação.
    """
    # Converte os rótulos para um array 2D (requisito do OneHotEncoder)
    rotulos = np.array(rotulos).reshape(-1, 1)

    # Cria uma instância do OneHotEncoder
    encoder = preprocessing.OneHotEncoder(sparse_output=False)  # sparse_output=False retorna um array denso

    # Ajusta o encoder aos rótulos e os transforma em OneHot
    rotulos_codificados = encoder.fit_transform(rotulos)
    print('Rótulos codificados com OneHotEncoder')
    return rotulos_codificados, encoder







    
