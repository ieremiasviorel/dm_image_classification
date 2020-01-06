import random
import re

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from definitions import IMAGES_DIR, ANNOTATIONS_DIR


class LocalizationDataset(Dataset):
    def __init__(self, file_folder, is_test=False, transform=None):
        self.img_folder_path = IMAGES_DIR
        self.annotation_folder_path = ANNOTATIONS_DIR
        self.file_folder = file_folder
        self.transform = transform
        self.is_test = is_test

    def __getitem__(self, idx):
        file = self.file_folder[idx]
        img_path = self.img_folder_path + '/' + file
        img = Image.open(img_path).convert('RGB')

        if not self.is_test:
            annotation_path = self.annotation_folder_path + '/' + file.split('.')[0]
            with open(annotation_path) as f:
                annotation = f.read()

            xy = self.get_xy(annotation)
            box = torch.FloatTensor(list(xy))

            new_box = self.box_resize(box, img)
            if self.transform is not None:
                img = self.transform(img)

            return img, new_box
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.file_folder)

    def get_xy(self, annotation):
        xmin = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', annotation)[0])
        xmax = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', annotation)[0])
        ymin = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', annotation)[0])
        ymax = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', annotation)[0])

        return xmin, ymin, xmax, ymax

    def show_box(self):
        file = random.choice(self.file_folder)
        annotation_path = self.annotation_folder_path + '/' + file.split('.')[0]

        img_box = Image.open(self.img_folder_path + '/' + file)
        with open(annotation_path) as f:
            annotation = f.read()

        draw = ImageDraw.Draw(img_box)
        xy = self.get_xy(annotation)
        print('bbox:', xy)
        draw.rectangle(xy=[xy[:2], xy[2:]])

        return img_box

    def box_resize(self, box, img, dims=(332, 332)):
        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        new_box = box / old_dims
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_box = new_box * new_dims

        return new_box
