import os
import json
from pathlib import Path
import copy
import urllib

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from Detections import Detection
from DataProcessing import Letterbox
from Misc import SaveBin, LoadBin

class COCO(Dataset):
    def __init__(self, image_dir, annotation_path, class_mapper):
        self.image_dir = image_dir

        # Load annotations
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        # Index images by ID
        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        annotations = data["annotations"]

        # Initialize ClassMapper
        self.class_mapper = class_mapper

        # Register categories
        for cat in sorted(data["categories"], key=lambda c: c["id"]):
            self.class_mapper.Register(cat["id"], cat["name"])

        # Map image IDs to their annotations
        image_to_anns = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in image_to_anns: image_to_anns[img_id] = []
            image_to_anns[img_id].append(ann)

        # Convert annotations to Detections
        self.detections = dict()
        for img_id, anns in image_to_anns.items():
            detections = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                class_id = self.class_mapper.LabelToClass(ann["category_id"])
                detection = Detection.from_topleft_xywh(x, y, w, h, class_id, 1.0)
                detections.append(detection)
            self.detections[img_id] = detections

        self.image_ids = list(self.detections.keys())
        self.num_classes = len(self.class_mapper)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.images[img_id])
        image = Image.open(img_path).convert("RGB")
        detections = self.detections[img_id]
        return image, detections
    
    @staticmethod
    def download(path, 
                 train_url = "http://images.cocodataset.org/zips/train2017.zip", 
                 val_url = "http://images.cocodataset.org/zips/val2017.zip", 
                 annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"):
      path = Path(path)
      train_path = path / "train.zip"
      val_path = path / "val.zip"
      anno_path = path / "annotations_trainval2017.zip"
      os.makedirs(path, exist_ok=True)

      for url, p in zip([train_url, val_url, annotations_url], [train_path, val_path, anno_path]):
        if not p.exists():
          print(f"[i] Downloading {url} ...")
          with urllib.request.urlopen(url) as connection:
            data = connection.read()
            with open(p, "wb") as f:
              f.write(data)
          print(f"[✓] Saved to {p}")
      
      return train_path, val_path, anno_path

class SimpleDataset(Dataset):
    def __init__(self, data, root_path): # [(image_name, dets), ...]
        self.data = data
        self.root_path = root_path
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_name, detections = self.data[idx]
        image_path = os.path.join(self.root_path, image_name)
        image = Image.open(image_path).convert("RGB")
        return image, detections

class DetectionDataset(Dataset):
    def __init__(self,
                 dataset,
                 class_mapper,
                 image_size=(640, 480), # Desired image size. (W, H)
                 augmentation = None, # Must take in pillow image and return pillow image. Can't resize, rotate, translate nor flip the image (or do anything else that would mess up the bounding boxes)
                 horizontal_flip = None, # Can be either None or single number (chance)
                 vertical_flip = None, # Can be either None or single number (chance)
                 rotate = None, # Angle in degrees, can be None, single number (Random(-x,x)) or two numbers (Random(x1, x2))
                 translate_w = None, # Translation given in percentage (0-1). Can be None, single number (Random(-x,x)) or two numbers (Random(x1, x2))
                 translate_h = None, # Translation given in percentage (0-1). Can be None, single number (Random(-x,x)) or two numbers (Random(x1, x2))
                 generate_once = False, # Use only for validation set. Upon creation it stores images in memory and it never augments them again.
                 device = "cpu", # Used by build-in image transformers
                 fill = (114, 114, 114), # Used by build-in image transformers
                 ):
        self.dataset = dataset
        self.image_size = image_size
        self.augmentation = copy.deepcopy(augmentation)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate = rotate
        self.translate_w = translate_w
        self.translate_h = translate_h
        self.stats = None # stats[class_id] = count
        self.class_mapper = class_mapper
        self.transformer = Letterbox(image_size, fill=fill, device = device)
        self.fill = fill
        self.__pregenerated = self.__pregenerate() if generate_once else None

    def __get_augmented_data(self, idx):
        image, raw_detections = self.dataset[idx]

        # Roll a dice for flips, rotations and so on. Must be the same for image and detections
        do_flip_h = self.horizontal_flip is not None and np.random.rand() < self.horizontal_flip
        do_flip_v = self.vertical_flip is not None and np.random.rand() < self.vertical_flip
        angle = (
            np.random.randint(-self.rotate, self.rotate)
            if isinstance(self.rotate, int) else
            np.random.uniform(*self.rotate)
            if isinstance(self.rotate, tuple) else
            None
        )
        dx = (
            np.random.uniform(-self.translate_w, self.translate_w)
            if isinstance(self.translate_w, float) else
            np.random.uniform(*self.translate_w)
            if isinstance(self.translate_w, tuple) else
            0
        )
        dy = (
            np.random.uniform(-self.translate_h, self.translate_h)
            if isinstance(self.translate_h, float) else
            np.random.uniform(*self.translate_h)
            if isinstance(self.translate_h, tuple) else
            0
        )

        # Original size of the image and size required by code - will be relevant later
        orig_W, orig_H = image.size
        W, H = self.image_size
        if dx: dx = dx * W
        if dy: dy = dy * H

        # Processing image
        #image = image.resize(self.image_size, Image.BILINEAR)  # (W, H)
        if self.augmentation: image = self.augmentation(image)
        image = self.transformer(image)
        if do_flip_h: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if do_flip_v: image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if angle: image = image.rotate(-angle, expand=False, fillcolor=self.fill)
        image = image.transform(image.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=self.fill)

        # Processing detections
        detections = []
        for detection in raw_detections:
          #detection = detection.Rescale((orig_W, orig_H), (W, H))
          detection = self.transformer.transform_detections( [detection], (orig_W, orig_H) )[0]
          if do_flip_h:
            detection = detection.HorizontalFlip(W)
          if do_flip_v:
            detection = detection.VerticalFlip(H)
          if angle:
            detection = detection.Rotate(angle, (W/2, H/2))
          if dx:
            detection = detection.TranslateW(dx)
          if dy:
            detection = detection.TranslateH(dy)
          detections.append( detection )

        return image, detections

    def __pregenerate(self):
        data = dict()
        for idx in range(len(self)):
            image, detections = self.__get_augmented_data(idx)
            data[idx] = (image, detections)
        return data

    def GetClassStats(self):
        if self.stats is None:
            self.stats = self.__compute_stats()
        return self.stats

    def __compute_stats(self):
        stats = {}
        for i in range(len(self.dataset)):
            _, detections = self.dataset[i]
            for detection in detections:
                cls = detection.class_id
                stats[cls] = stats[cls] + 1 if cls in stats else 1
        return stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Special case: only pregenerated images
        if self.__pregenerated:
          return self.__pregenerated[idx]
        return self.__get_augmented_data(idx)

    def SaveAugmentedDataset(self, root_path, extension = ".png") -> None:
      os.makedirs(root_path, exist_ok=True)
      indexes = list(range(len(self)))
      data = []
      for idx in indexes:
        image_name = f"{idx}{extension}"
        image_path = os.path.join(root_path, image_name)
        image, dets = self[idx]
        data.append( (image_name, dets) )
        image.save(image_path)
      SaveBin(os.path.join(root_path, "data.pkl"), data)

    @staticmethod
    def LoadAugmentedDataset(root_path) -> SimpleDataset:
      data = LoadBin(os.path.join(root_path, "data.pkl"))
      return SimpleDataset(data, root_path)