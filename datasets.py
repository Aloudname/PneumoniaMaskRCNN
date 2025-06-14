import os
import cv2
import json
import glob
import tqdm
import torch
import shutil
import imgviz
import argparse
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from labelme import utils
from pycocotools.coco import COCO
import matplotlib.patches as patches
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CocoDetection
from sklearn.model_selection import train_test_split


np.random.seed(41)
 
# Spare 0 for background. Attention!
# classname_to_id = {"非典型增生": 1, "非典型增生术后": 2, "口腔白斑": 3, "口腔白斑术后": 4,
#              "口腔红斑": 5, "口腔黏膜溃痛": 6, "口腔苔藓化损伤": 7, "口腔苔藓样损伤": 8, 
#              "鳞状细胞癌术后": 9, "糜烂": 10, "疣状黄瘤": 11, "肿物": 12, "CA术后": 13}

classname_to_id = {"cancer_area": 1}

class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
 
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent = 2 for more delicate display.

    # Transform .json into COCO dataset.
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance
 
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)
 
    # COCO.image
    def _image(self, obj, path):
        image = {}
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image
 
    # COCO.annotation.
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation
 
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
 
    # COCO format [x1,y1,w,h] aligns with COCO bbox.
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

class CocoDataset(CocoDetection):
    """
    Inherited from torchvision.CocoDetection dataset format,
    with all belonging functions.
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file, transforms=None)
        self._load_coco_annotations(ann_file)
        self.c_transforms = transforms

    def _load_coco_annotations(self, ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        self.coco = COCO()
        self.coco.dataset = annotations
        self.coco.createIndex()

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

    # filter for invalid box.
        valid_annotations = []
        for ann in target:

            bbox = ann['bbox']
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                valid_annotations.append(ann)
    
        if not valid_annotations:
            bboxes = []
            labels = []
            masks = []
        else:
            bboxes = [ann['bbox'] for ann in valid_annotations]
            labels = [ann['category_id'] for ann in valid_annotations]
        
            masks = []
            for ann in valid_annotations:
                mask = self.coco.annToMask(ann)
                masks.append(mask)
    

        if masks:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        else:
            # Null mask code.
            w, h = img.size
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
    
        img_np = np.array(img.convert('RGB'))
    
        if self.c_transforms is not None:
            try:
                transformed = self.c_transforms(
                    image=img_np,
                    masks=masks.numpy() if masks.numel() > 0 else [],
                    bboxes=bboxes,
                    category_ids=labels
                )
                img = transformed['image']
            
                masks = transformed['masks']
                bboxes = transformed['bboxes']
                labels = transformed['category_ids']
            except Exception as e:
                print(f"Error during augmentation for image {image_id}: {str(e)}")
                pass
    
        if bboxes:
            boxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
            if masks and isinstance(masks, list):
                masks_tensor = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks], dim=0)
            else:
                w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
                masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            # create null vector if none valid box.
            w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
    
        target = {
            'image_id': torch.tensor([image_id], dtype=torch.int64),
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'masks': masks_tensor,
        }
    
        return img, target

    def _sample(self, case_id = 0, frame_id = 0, ann_file = '/home/sdu/BME2025chenhaoran/Mask-R-CNN-with-Resnet101/input/annotations/instances_train.json', show_image=True):
        """
        Validate with highly customizable parameters in COCO dataset.
        
        params:
        ann_file: annotation file path.
        case_id: sample ID (int, e.g. 0 -> training_000)
        frame_id: frame ID in a given sample (int, e.g. 31 -> training_XXX_031)
        show_image: show img or label area in plot.
        """
        case_str = f"training_{case_id:03d}"
        expected_filename = f"{case_str}_{frame_id:03d}.png"
        
        # check the image file.
        image_path = os.path.join(self.root, expected_filename)
        if not os.path.exists(image_path):
            print(f"❌ No image file: {image_path}")
            return False
    
        print(f"✅ Image file exists: {image_path}")

        if not os.path.exists(ann_file):
            print(f"❌ No annotation file: {ann_file}")
            return False
    
        coco = COCO(ann_file)
    
        # check a single frame.
        img_id = None
        for img_info in coco.dataset["images"]:
            if img_info["file_name"] == expected_filename:
                img_id = img_info["id"]
                print(f"✅ Target image exist: (ID: {img_id})")
                break
    
        if img_id is None:
            print(f"❌ No image target in sample: {expected_filename}")
            return False
        
        # check the annotation. 
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
    
        if not annotations:
            print("⚠️ Found target but without annotation!")
            return False
    
        print(f"✅ Found {len(annotations)} targets.")
    
        # check the mask. 
        valid_annotations = 0
        for ann in annotations:
            if "segmentation" in ann and ann["segmentation"]:
                # Check mask code in RLE.
                if isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]:
                    rle = ann["segmentation"]
                    mask = coco.annToMask(ann)
                    area = np.sum(mask)
                
                    # confirm the area of targeted and caculated area.
                    if abs(area - ann["area"]) > 1:
                        print(f"⚠️ Targeted ID {ann['id']} differs between caculated value and targeted value: "
                              f"caculated={area:.2f}, targeted={ann['area']:.2f}")
                    else:
                        print(f"✅ Targeted ID {ann['id']} RLE mask area : {ann['area']:.2f}")
                        valid_annotations += 1
            
                bbox = ann["bbox"]
                if bbox[2] > 0 and bbox[3] > 0:
                    print(f"  Boundary box: x={bbox[0]:.1f}, y={bbox[1]:.1f}, "
                          f"width={bbox[2]:.1f}, height={bbox[3]:.1f}")
                else:
                    print(f"⚠️ Targeted ID {ann['id']} invalid boundary box: {bbox}")
    
        # visualization.
        if show_image and valid_annotations > 0:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img, cmap='gray')
            plt.title(f"Case {case_str} | Frame {frame_id}")
        
            for ann in annotations:
                bbox = ann["bbox"]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                plt.gca().add_patch(rect)
            
                mask = coco.annToMask(ann)
                mask = np.ma.masked_where(mask == 0, mask)
                plt.imshow(mask, alpha=0.5, cmap='jet')
        
            plt.axis('off')
            plt.show()
    
        return valid_annotations > 0

# Below are selectable transforms.
# Before changing library of transforms, 
# do remember to adjust CocoDataset.__getitem__() into suitable structure.
# Do note that also convert mask code into tensors.

# transform = T.ToTensor()

# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor()])

transform = A.Compose([
    A.Resize(256, 256),
        # A.ToRGB(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()],
    bbox_params = A.BboxParams(
        format = 'coco',
        # min_visibility=0.1, # if required to filt very small frames.
        label_fields = ['category_ids']))


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

def Get_mask(annotation_dir, img_dir):
    """
    Get the mask from an image.
    Run ONLY ONCE when masks are required.
    masks will be saved in .png format.

    params:
    annotation_dir: annotations directory.
    split: train or test directory.
    """
    annotation_file = os.path.join(annotation_dir)

    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))

    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            mask_path = os.path.join(img_dir, img['file_name'].replace('.jpg', '.png'))
            save_colored_mask(mask, mask_path)


def Get_COCO(labelme_path, saved_coco_path):
    print('reading...')
    # mkdir.
    if not os.path.exists("%scoco\\annotations" % saved_coco_path):
        os.makedirs("%scoco\\annotations" % saved_coco_path)

    if not os.path.exists("%scoco\\images\\train" % saved_coco_path):
        os.makedirs("%scoco\\images\\train" % saved_coco_path)

    if not os.path.exists("%scoco\\images\\test" % saved_coco_path):
        os.makedirs("%scoco\\images\\test" % saved_coco_path)

    print(labelme_path + "\*.json")
    json_list_path = glob.glob(labelme_path + "\*.json")
    # json_list_path = glob.glob(labelme_path + "\*.png")
    print('json_list_path: ', len(json_list_path))
    # data split without distinguish between train and test.
    train_path, val_path = train_test_split(json_list_path, test_size=0.2, train_size=0.8)
    print(f"train samples: {len(train_path)}, test samples: {len(val_path)}.")

    # Transform train .json into COCO dataset.
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco\\annotations\\instances_train.json' % saved_coco_path)

    for file in train_path:
        # print("Testing: file："+file)
        img_name = file.replace('json', 'png')
        # print("Testing: img_name：" + img_name)
        temp_img = cv2.imread(img_name)
        # if None, img is .jpg format.
        if  temp_img is None:
            img_name_jpg = img_name.replace('png', 'jpg')
            temp_img = cv2.imread(img_name_jpg)
 
        filenames = img_name.split("\\")[-1]
        cv2.imwrite("E:\\kq\\input\\coco\\images\\train\\{}".format(filenames), temp_img)
        # print(temp_img)

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco\\images\\test\\" % saved_coco_path)
 
        img_name = file.replace('json', 'png')
        temp_img = cv2.imread(img_name)
        if temp_img is None:
            img_name_jpg = img_name.replace('png', 'jpg')
            temp_img = cv2.imread(img_name_jpg)
        filenames = img_name.split("\\")[-1]
        cv2.imwrite("E:\\kq\\dataset\\coco\\images\\test\\{}".format(filenames), temp_img)
 
    # Transform train .json into COCO dataset.
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco\\annotations\\instances_test.json' % saved_coco_path)
    print("COCO dataset generated successfully!")

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        # Convert to 3d tensor [C, H, W].
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        images.append(img)
        targets.append(target)
    return images, targets

# finally we get coco-style datasets.
train_loader = CocoDataset('/home/sdu/BME2025chenhaoran/Mask-R-CNN-with-Resnet101/input/images/train', '/home/sdu/BME2025chenhaoran/Mask-R-CNN-with-Resnet101/input/annotations/instances_train.json', transforms = transform)
test_loader = CocoDataset('/home/sdu/BME2025chenhaoran/Mask-R-CNN-with-Resnet101/input/images/test', '/home/sdu/BME2025chenhaoran/Mask-R-CNN-with-Resnet101/input/annotations/instances_test.json', transforms = transform)

# check the availability with highly customizable parameters.

if __name__ == '__main__':

    labelme_path = "E:\\kq\\input\\test"
    saved_coco_path = "E:\\kq\\input\\test"

    print('torch version: ' + torch.__version__)
    print('torchvision version: ' + torchvision.__version__)
    print('if cuda available: ' + str(torch.cuda.is_available()))
    # # run only ONCE when first generates a coco directory.
    # Get_COCO(labelme_path, saved_coco_path)
    # Get_mask('E:\\kq\\input\\coco\\annotations\\instances_test.json', 'E:\\kq\\input\\coco\\images\\test')
    # Get_mask('E:\\kq\\input\\coco\\annotations\\instances_train.json', 'E:\\kq\\input\\coco\\images\\train')

