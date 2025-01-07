from trim_dataset import TrimDataset
import matplotlib.pyplot as plt
import random, cv2

import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
import random

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




dataset = TrimDataset('normalized_dataset')
plot_class_grid(dataset,5, 'gt_mask')