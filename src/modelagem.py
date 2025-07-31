from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def criar_modelo(img_width : int, img_height : int, channels : int, num_classes : int) -> Sequential:
    """
    Cria e compila o modelo de rede neural convolucional para reconhecimento de placas de veículos.
    Parametros:
    - img_width: Largura da imagem de entrada.
    - img_height: Altura da imagem de entrada.
    - channels: Número de canais da imagem (1 para grayscale, 3 para RGB).
    - num_classes: Número de classes (caracteres) a serem reconhecidos.
    Retorna:
    - model: O modelo compilado.
    """
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)),
        MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Achata a saída para camadas densas
        Flatten(),
        
        # Camadas densas
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularização para evitar overfitting
        
        # Camada de saída
        Dense(num_classes, activation='softmax')
    ])

    # Compila o modelo
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    print(model.summary())
    
    return model