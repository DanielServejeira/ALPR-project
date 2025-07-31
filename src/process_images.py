import cv2 as cv
import os
import numpy as np

def process_image(img_path : str, output_folder : str) -> None:
    """
    Processa uma imagem para melhorar a qualidade e salvar no diretório de saída.
    Parametros:
    img_path: Caminho da imagem a ser processada.
    output_folder: Pasta onde a imagem processada será salva.
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    inv = cv.bitwise_not(img)

    _, thresh = cv.threshold(inv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    filename = os.path.basename(img_path)
    output_path = os.path.join(output_folder, filename)

    cv.imwrite(output_path, opening)

def process_images() -> None:
    """
    Processa todas as imagens nas pastas de treino e validação, salvando os resultados na pasta de saída.
    """
    output_folder = 'output/'
    train_folder = 'train/'
    val_folder = 'val/'

    os.makedirs(os.path.join(output_folder, train_folder), exist_ok=True)
    os.makedirs(os.path.join(output_folder, val_folder), exist_ok=True)

    list_dirs = os.listdir(train_folder)
    for diretorio in list_dirs:
        char_folder = os.path.join(train_folder, diretorio)
        output_char_folder = os.path.join(output_folder, char_folder)
        os.makedirs(output_char_folder, exist_ok=True)

        list_arquivos = os.listdir(char_folder)
        for arquivo in list_arquivos:
            input_path = os.path.join(char_folder, arquivo)
            output_path = os.path.join(output_folder, char_folder)
            process_image(input_path, output_path)

        print(f"========== Processadas imagens da pasta {char_folder} ==========")

    list_dirs = os.listdir(val_folder)
    for diretorio in list_dirs:
        char_folder = os.path.join(val_folder, diretorio)
        output_char_folder = os.path.join(output_folder, char_folder)
        os.makedirs(output_char_folder, exist_ok=True)

        list_arquivos = os.listdir(char_folder)
        for arquivo in list_arquivos:
            input_path = os.path.join(char_folder, arquivo)
            output_path = os.path.join(output_folder, char_folder)
            process_image(input_path, output_path)

        print(f"========== Processadas imagens da pasta {char_folder} ==========")

    print("Imagens processadas salvas na pasta output")
