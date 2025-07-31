import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import Sequential

def avaliar_modelo(model : Sequential, X_test : np.ndarray, y_test : np.ndarray, history : dict) -> None:
    """
    Avalia o modelo treinado no conjunto de teste e plota a acurácia.
    Parametros:
    - model: modelo treinado.
    - X_test: dados de teste.
    - y_test: rótulos de teste.
    """
    # Avalia no conjunto de teste
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nAcurácia no teste: {test_acc:.4f}')

    plt.plot(history.history['accuracy'], label='Acurácia (treino)')
    plt.plot(history.history['val_accuracy'], label='Acurácia (validação)')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

def matriz_confusao(y_pred : np.ndarray, y_test : np.ndarray, num_classes : int) -> None:
    """
    Plota a matriz de confusão.
    Parametros:
    - y_pred: rótulos previstos pelo modelo.
    - y_test: rótulos reais.
    - num_classes: número de classes.
    """
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes), rotation=45)
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

def metricas_classificacao(y_test : np.ndarray, y_pred : np.ndarray) -> None:
    """
    Calcula e exibe as métricas de classificação: precisão, recall e F1-score.
    Parametros:
    - y_test: rótulos reais.
    - y_pred: rótulos previstos pelo modelo.
    """
    letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average=None)

    for i in range(10):
        print(f"\nPrecisão para o número {i}:", precision[i])
        print(f"Recall para o número {i}:", recall[i])
        print(f"F1 Score para o número {i}:", f1_score[i])

    for i in range(len(letras)):
        print(f"\nPrecisão para letra {letras[i]}:", precision[10 + i])
        print(f"Recall para letra {letras[i]}:", recall[10 + i])
        print(f"F1 Score para letra {letras[i]}:", f1_score[10 + i])