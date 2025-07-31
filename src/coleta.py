import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
import time

def load_images(folder_path : str, label: int, channels: int, target_size: tuple = (75, 100), img_width: int = 75, img_height: int = 100) -> tuple:
    """
    Carrega imagens de uma pasta, redimensiona e normaliza.
    Parâmetros:
    - folder_path: caminho da pasta com as imagens.
    - label: rótulo associado às imagens.
    - channels: número de canais da imagem (1 para grayscale, 3 para RGB).
    - target_size: tamanho para redimensionar as imagens.
    - img_width: largura da imagem.
    - img_height: altura da imagem.
    Retorna:
    - images: array de imagens normalizadas.
    - labels: array de rótulos correspondentes.
    """
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(folder_path, filename))
                # Converte para RGB para garantir 3 canais
                img = img.convert('RGB').resize(target_size)
                img_array = np.array(img) / 255.0
                
                # Verifica se a imagem tem o shape correto
                if img_array.shape == (img_height, img_width, channels):
                    images.append(img_array)
                    labels.append(label)
                else:
                    print(f"Imagem com shape inválido: {img_array.shape} - {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {str(e)}")
    
    return np.array(images), np.array(labels)

def coletar_dados(folder_path : str, channels: int, num_classes: int, img_width: int, img_height: int) -> tuple:
    """
    Coleta dados de imagens e rótulos de treino e validação.
    Parâmetros:
    - folder_path: caminho da pasta com as imagens.
    - channels: número de canais da imagem (1 para grayscale, 3 para RGB).
    - num_classes: número total de classes (10 dígitos + 26 letras).
    - img_width: largura da imagem.
    - img_height: altura da imagem.
    Retorna:
    - X_train: array de imagens de treino.
    - y_train: array de rótulos de treino.
    - X_test: array de imagens de validação.
    - y_test: array de rótulos de validação.
    """
    # Lista de letras
    letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Carrega dados - TREINO
    X_train, y_train = load_images(os.path.join(folder_path, "output/train/0"), 0, channels, (img_width, img_height), img_width, img_height)

    # Carrega dados - VALIDAÇÃO
    X_test, y_test = load_images(os.path.join(folder_path, "output/val/0"), 0, channels, (img_width, img_height), img_width, img_height)

    # Carrega dígitos 1-9
    for i in range(1, 10):
        # Treino
        X_tr, y_tr = load_images(os.path.join(folder_path, f"output/train/{i}"), i, channels, (img_width, img_height), img_width, img_height)
        X_train = np.concatenate((X_train, X_tr))
        y_train = np.concatenate((y_train, y_tr))
        
        # Validação
        X_te, y_te = load_images(os.path.join(folder_path, f"output/val/{i}"), i, channels, (img_width, img_height), img_width, img_height)
        X_test = np.concatenate((X_test, X_te))
        y_test = np.concatenate((y_test, y_te))

    # Carrega letras
    for letra in letras:
        label = 10 + letras.index(letra)
        # Treino
        X_tr, y_tr = load_images(os.path.join(folder_path, f"output/train/{letra}"), label, channels, (img_width, img_height), img_width, img_height)
        X_train = np.concatenate((X_train, X_tr))
        y_train = np.concatenate((y_train, y_tr))
        
        # Validação
        X_te, y_te = load_images(os.path.join(folder_path, f"output/val/{letra}"), label, channels, (img_width, img_height), img_width, img_height)
        X_test = np.concatenate((X_test, X_te))
        y_test = np.concatenate((y_test, y_te))

    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, y_train, X_test, y_test