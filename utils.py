from trim_dataset import TrimDataset
import matplotlib.pyplot as plt
import random, cv2, os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.patches as patches
import numpy as np

def plot_class_grid(trim_dataset, num_samples=10, mode="nobbox"):
    """
    Plota um grid contendo num_samples imagens, uma de cada classe, com opção de mostrar bounding boxes ou máscaras.
    
    Args:
        trim_dataset (TrimDataset): Objeto do dataset.
        num_samples (int): Número de imagens para exibir.
        mode (str): Define o modo de visualização: 
                    "bbox" para imagens com bounding boxes,
                    "nobbox" para imagens sem bounding boxes,
                    "gt_mask" para imagens com suas máscaras ground truth.
    """
    if mode not in ["bbox", "nobbox", "gt_mask"]:
        raise ValueError("O parâmetro 'mode' deve ser 'bbox', 'nobbox' ou 'gt_mask'.")

    categories = trim_dataset.categories
    category_ids = list(categories.keys())
    
    fig, axes = plt.subplots(num_samples, len(category_ids), figsize=(len(category_ids) * 2, num_samples * 2))
    
    for col, cat_id in enumerate(category_ids):
        # Filtrar imagens da classe atual
        images = [img for img in trim_dataset.images if img.category_id == cat_id]
        if len(images) < num_samples:
            print(f"A classe {categories[cat_id]} possui menos de {num_samples} imagens.")
            sampled_images = images  # Mostrar todas as disponíveis
        else:
            sampled_images = random.sample(images, num_samples)
        
        # Adicionar título da classe na primeira linha
        axes[0, col].set_title(categories[cat_id], fontsize=10)
        
        # Preencher a coluna com as imagens selecionadas
        for row, img in enumerate(sampled_images):
            ax = axes[row, col]
            if mode == "bbox":
                # Desenhar bounding box na imagem
                img_with_bbox = img.content.copy()
                h, w = img_with_bbox.shape[:2]
                x_min, y_min, box_w, box_h = img.bbox
                x_max = int(x_min + box_w)
                y_max = int(y_min + box_h)
                
                # Ajustar para a escala da imagem (caso necessário)
                x_min = int(x_min * (w / img.content.shape[1]))
                x_max = int(x_max * (w / img.content.shape[1]))
                y_min = int(y_min * (h / img.content.shape[0]))
                y_max = int(y_max * (h / img.content.shape[0]))
                
                cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Azul
                ax.imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
            
            elif mode == "nobbox":
                # Mostrar imagem sem bounding box
                ax.imshow(cv2.cvtColor(img.content, cv2.COLOR_BGR2RGB))
            
            elif mode == "gt_mask":
                # Mostrar máscara ground truth
                ax.imshow(img.mask, cmap="gray")
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_cnn_with_bboxes(input_shape, num_classes)->models.Model:
    """
    Cria uma CNN para classificação e regressão de bounding boxes.
    
    :param input_shape: Dimensões da imagem de entrada (altura, largura, canais).
    :param num_classes: Número de classes para a classificação.
    :return: Modelo CNN compilado.
    """
    inputs = tf.keras.Input(shape=input_shape)

    kernel = (5, 5)
    pool = (2, 2)


    x = layers.Conv2D(16, kernel, activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloco convolucional 1
    x = layers.Conv2D(32, kernel, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloco convolucional 2
    x = layers.Conv2D(64, kernel, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloco convolucional 3
    #x = layers.Conv2D(128, kernel, activation='relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)

    # Flatten para conexão densa
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    # Saída para classificação
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)

    # Saída para bounding boxes
    bbox_output = layers.Dense(4, name='bbox_output')(x)

    # Modelo com duas saídas
    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])

    # Compilação do modelo
    model.compile(
        optimizer='adam',
        loss={
            'class_output': 'categorical_crossentropy',
            'bbox_output': 'mean_squared_error'
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mean_squared_error'
        }
    )

    return model

def data_to_dict(image, bbox, class_id):
    return {"image":image, "bbox": bbox,"class":class_id}

def preprocess_images(images):
    """
    Pré-processa uma lista de imagens lidas com cv2.imread para uso em TensorFlow.
    :param images: Lista de imagens no formato BGR (uint8) lidas com cv2.
    :return: Imagens normalizadas e convertidas para RGB, no formato float32.
    """
    preprocessed_images = []
    for img in images:
        # Verifica se a imagem é válida
        if img is None:
            raise ValueError("Uma ou mais imagens no dataset são inválidas.")
        
        # Converte de BGR para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normaliza os valores para o intervalo [0, 1]
        img_normalized = img_rgb / 255.0
        
        # Converte para float32
        img_float32 = img_normalized.astype(np.float32)
        
        preprocessed_images.append(img_float32)
    
    return np.array(preprocessed_images)

def split_dataset(data, train_ratio=0.8, test_ratio=0.1, validate_ratio=0.1):
    """
    Divide um conjunto de dados (vetor de dicionários) em três subconjuntos: treino, teste e validação,
    com base nas proporções fornecidas.

    :param data: Lista de dicionários contendo os dados.
    :param train_ratio: Proporção dos dados para o conjunto de treino (default: 0.8).
    :param test_ratio: Proporção dos dados para o conjunto de teste (default: 0.1).
    :param validate_ratio: Proporção dos dados para o conjunto de validação (default: 0.1).
    :return: Três listas: (train_data, test_data, validate_data)
    """
    
    # Verificar se as proporções somam 1
    if train_ratio + test_ratio + validate_ratio != 1:
        raise ValueError("As proporções de treino, teste e validação devem somar 1.")

    # Embaralha aleatoriamente os dados
    random.shuffle(data)

    # Calcula os índices para separar os dados
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)

    # Divide os dados
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    validate_data = data[train_size + test_size:]

    return train_data, test_data, validate_data