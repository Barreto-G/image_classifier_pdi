import json, os, cv2
from image import Image
import albumentations as A
import uuid

class TrimDataset:
    def __init__(self, base_dir, json_file:str):
        self.base_dir = base_dir
        self.images = []        
        self.categories = {}
        self.json = None

        self._load_data(json_file)

    def _load_data(self, json_file):
        try:
            with open(json_file) as f:
                self.json = json.load(f)
        except FileNotFoundError:
            print("Erro: Arquivo nao encontrado no caminho especificado")
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar arquivo JSON: {e}")

        self.categories = {cat['id']:cat['name'] for cat in self.json['categories']}

        for img_data in self.json['images']:
            img = Image()
            ann = self.json[''][img_data['id']-1]

            img.category_id = ann['category_id']

            category_name = self.categories[img.category_id]
            img_name = img_data['file_name']
            img_path = os.path.join(self.base_dir, category_name, img_name)
            img.content = cv2.imread(img_path)
            if img.content is None:
                print(f"Erro ao ler imagem no caminho: {img_path}")

            rle_mask = ann['segmentation']['counts']
            img.mask = img._rle_to_mask(rle_mask, (img_data['height'], img_data['width']))
            img.bbox = ann['bbox']
            img.file_name = img_name
            self.images.append(img)

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
                augmented_img_name = f'{image.file_name}-augmented-{uuid.uuid4().hex[:4]}.png'
                aux = Image(name=augmented_img_name, 
                                      bbox=augmented_img['bbox'],
                                      mask= augmented_img['mask'],
                                      category=image.category_id,
                                      content=augmented_img['image'])
                self.images.append(aux)
                path_augmented = os.path.join('augmented_dataset', self.categories[aux.category_id])
                if not os.path.exists(path_augmented):
                    os.makedirs(path_augmented)
                path_augmented = os.path.join(path_augmented, aux.file_name)
                cv2.imwrite(path_augmented, aux.content)
                self.add_json(aux)

    def add_json(self, image: Image):
        segmentation = image._mask_to_rle(image.mask)
        img_id = len(self.images)
        annotation = {
            'id': img_id,
            'image_id': img_id,
            'category_id': image.category_id,
            'area': image.bbox[2] * image.bbox[3],
            'bbox': image.bbox,
            'iscrowd': 0,
            'segmentation': {'counts': segmentation, 'size': [2112, 2112]}
        }

        self.json[''].append(annotation)

        image_info = {
            'id': img_id,
            'file_name': image.file_name,
            'width': 2112,
            'height': 2112
        }

        self.json['images'].append(image_info)

    def save_annotations(self):
        os.makedirs('output', exist_ok=True)
        with open(os.path.join('output', 'annotations.json'), 'w') as f:
            json.dump(self.json, f, indent=0)

    def equalize_histogram(self, image):
        pass

    def normalize_dataset(self):
        pass
    
    def __repr__(self):
        return f'base_dir:{self.base_dir},categories:{self.categories},images:{[image for image in self.images]}'

if __name__ == "__main__":
    json_data_path = 'segmentation_data.json'
    images_path = 'trims_dataset'

    dataset = TrimDataset(images_path, json_data_path)
    dataset.dataset_augmentation()
    dataset.save_annotations()