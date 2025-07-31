import warnings
warnings.filterwarnings("ignore")

import src.avaliar as av
import src.coleta as coleta
import src.modelagem as modelagem
import src.process_images as process_images
import time
import sys

def main(img_width : int, img_height : int, channels : int, num_classes : int, folder_path : str) -> int:
    start = time.time()
    
    if 'processar_imagens' in sys.argv[1:]:
        process_images.process_images()

    X_train, y_train, X_test, y_test = coleta.coletar_dados(folder_path, channels, num_classes, img_width, img_height)

    model = modelagem.criar_modelo(img_width, img_height, channels, num_classes)

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    av.avaliar_modelo(model, X_test, y_test, history)

    y_pred = model.predict(X_test)

    av.matriz_confusao(y_pred, y_test, num_classes)

    av.metricas_classificacao(y_test, y_pred)

    end = time.time()

    return int(end - start)

if __name__ == '__main__':
    img_width, img_height = 75, 100
    channels = 3
    num_classes = 35

    # MUDAR O CAMINHO PARA A PASTA ONDE ESTA O PROJETO
    folder_path = "C:/Users/VitorRodrigues/ALPR-project/"

    execution_time = main(img_width, img_height, channels, num_classes, folder_path)

    print(f"Tempo de execução: {execution_time} segundos")
