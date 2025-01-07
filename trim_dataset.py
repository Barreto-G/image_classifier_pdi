import json, os, cv2
from image import Image
import albumentations as A
import numpy as np

class TrimDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.images = []        
        self.categories = {}
        self.json = None

        self._load_data()

    def _load_data(self):
        try:
            with open(os.path.join(self.base_dir, 'metadata.json'), 'r') as f:
                self.json = json.load(f)
        except FileNotFoundError:
            print("Erro: Arquivo nao encontrado no caminho especificado")
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar arquivo JSON: {e}")

        self.categories = {cat['id']:cat['name'] for cat in self.json['categories']}

        for img_data in self.json['images']:
            img = Image()
            ann = self.json['annotation'][img_data['id']-1]

            img.category_id = ann['category_id']

            category_name = self.categories[img.category_id]
            img_name = img_data['file_name']
            img_path = os.path.join(self.base_dir, category_name, img_name)
            img.content = cv2.imread(img_path)
            if img.content is None:
                print(f"Erro ao ler imagem no caminho: {img_path}")
                print(f"ID da categoria:{img.category_id}; nome da categoria:{category_name}")
                print(f"IMG DATA:{img_data}")
                print(f"ANNOTATIONS:{ann}")
                continue
            else:
                rle_mask = ann['segmentation']['counts']
                img.mask = img._rle_to_mask(rle_mask, (img_data['height'], img_data['width']))
                img.bbox = ann['bbox']
                img.file_name = img_name

                if img.content.shape[:2] == [224,224]:
                    self.images.append(img)
                else:
                    # Redimensionando as fotos
                    resized_image = self.resize_image(img)
                    self.images.append(resized_image)
                    img_data['width'] = resized_image.content.shape[0]
                    img_data['height'] = resized_image.content.shape[1]
                    ann['size'] = resized_image.content.shape[:2]
                    ann['bbox'] = resized_image.bbox
                    ann['area'] = resized_image.bbox[2] * resized_image.bbox[3]
                    ann['segmentation']['counts'] = resized_image._mask_to_rle(resized_image.mask)
            

    def _img_augmentation(self, image, angle=15):
        transform = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=angle, border_mode=cv2.BORDER_REPLICATE),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.5),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0,
                                 sat_shift_limit=10, val_shift_limit=0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        result = transform(image=image.content, bboxes=[image.bbox], mask=image.mask, labels=[0])
        result['bbox'] = [float(round(b)) for b in result['bboxes'][0]]
        return result

    def dataset_augmentation(self):
        copy_images = self.images.copy()
        for image in copy_images:
            for i in range(2):
                augmented_img = self._img_augmentation(image)
                augmented_img_name = f'{image.file_name[:-4]}-augmented-{i}.png'
                aux = Image(name=augmented_img_name, 
                                      bbox=augmented_img['bbox'],
                                      mask= augmented_img['mask'],
                                      category=image.category_id,
                                      content=augmented_img['image'])
                self.images.append(aux)
                self.add_json(aux)
        self.save_images('augmented_dataset')
        self.save_annotations('augmented_dataset')

    def add_json(self, image: Image, json_file= None):
        segmentation = image._mask_to_rle(image.mask)
        if json_file == None:
            img_id = len(self.images)
        else:
            img_id = len(json_file['images'])+1

        annotation = {
            'id': img_id,
            'image_id': img_id,
            'category_id': image.category_id,
            'area': image.bbox[2] * image.bbox[3],
            'bbox': image.bbox,
            'iscrowd': 0,
            'segmentation': {'counts': segmentation, 'size': [image.content.shape[0], image.content.shape[1]]}
        }

        image_info = {
            'id': img_id,
            'file_name': image.file_name,
            'width': image.content.shape[0],
            'height': image.content.shape[0]
        }
        if json_file == None:
            self.json['annotation'].append(annotation)
            self.json['images'].append(image_info)
        else:
            json_file['annotation'].append(annotation)
            json_file['images'].append(image_info)

    def equalize_histogram(self, image):
        img_yuv = cv2.cvtColor(image.content, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(
            clipLimit=1.0, tileGridSize=(8,8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def normalize_dataset(self):
        normalized_images = []
        normalized_json = self.json.copy()
        normalized_json['annotation'].clear()
        normalized_json['images'].clear()
        for image in self.images:
            equalized_img = self.equalize_histogram(image)
            equalized_img_name = f'{image.file_name[:-4]}-equalized.png'
            aux = Image(name=equalized_img_name, 
                                    bbox=image.bbox,
                                    mask= image.mask,
                                    category=image.category_id,
                                    content=equalized_img)
            normalized_images.append(aux)
            self.add_json(aux, normalized_json)
        self.save_images('normalized_dataset', normalized_images)
        self.save_annotations(path='normalized_dataset', json_file=normalized_json)
    
    def save_annotations(self, path="", file_name="metadata.json", json_file=None):
        json_aux = json_file if json_file != None else self.json
        if path != "":
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, file_name), 'w') as f:
                json.dump(json_aux, f, indent=None)
        else:
            with open(file_name, 'w') as f:
                json.dump(json_aux, f, indent=None)    

    def save_images(self, data_folder, images = None):
        if images is None:
            images = self.images
        for img in images:
            path = os.path.join(data_folder, self.categories[img.category_id])
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, img.file_name)
            cv2.imwrite(path, img.content)
    
    def split_dataset(self, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1):
        num_images = len(self.images)
        num_train = int(train_ratio * num_images)
        num_val = int(eval_ratio * num_images)

        indices = np.random.permutation(num_images)
        self.data_split['train'] = indices[:num_train]
        self.data_split['val'] = indices[num_train:num_train + num_val]
        self.data_split['test'] = indices[num_train + num_val:]


    def resize_image(self, image:Image, target_size=(224, 224)):
        """
        Redimensiona uma imagem e ajusta sua bounding box proporcionalmente ao novo tamanho.
        
        :param image: Imagem original (lida com cv2.imread()).
        :param bbox: Bounding box original no formato [x_min, y_min, x_max, y_max].
        :param target_size: Tamanho de destino da imagem (largura, altura).
        :return: Imagem redimensionada e bounding box ajustada.
        """
        # Obtém as dimensões originais da imagem
        original_height, original_width = image.content.shape[:2]
        target_width, target_height = target_size

        # Redimensiona a imagem
        resized_image = cv2.resize(image.content, (target_width, target_height))
        resized_mask = cv2.resize(image.mask, (target_width, target_height))

        # Calcula os fatores de escala
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Ajusta as coordenadas da bounding box
        x_min, y_min, x_max, y_max = image.bbox
        x_min_new = int(x_min * scale_x)
        y_min_new = int(y_min * scale_y)
        x_max_new = int(x_max * scale_x)
        y_max_new = int(y_max * scale_y)

        resized_bbox = [x_min_new, y_min_new, x_max_new, y_max_new]

        return Image(name=image.file_name, bbox=resized_bbox, category=image.category_id, content=resized_image, mask=resized_mask)

    def __repr__(self):
        return f'base_dir:{self.base_dir},categories:{self.categories},images:{[image for image in self.images]}'

if __name__ == "__main__":
    print("Trim Dataset definition")