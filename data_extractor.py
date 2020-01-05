import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from definitions import get_absolute_path

breed_list = os.listdir(get_absolute_path("input/stanford-dogs-dataset/images/Images/"))


def label_assignment(img, label):
    return label


X = []
Z = []
imgsize = 150


def training_data(label, data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img, label)
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))

        X.append(np.array(img))
        Z.append(str(label))


def show_dir_images(breed, n_to_show):
    plt.figure(figsize=(16, 16))
    img_dir = get_absolute_path("input/stanford-dogs-dataset/images/Images/{}/".format(breed))
    images = os.listdir(img_dir)[:n_to_show]
    for ii in range(n_to_show):
        img = mpimg.imread(img_dir + images[ii])
        plt.subplot(n_to_show / 4 + 1, 4, ii + 1)
        plt.imshow(img)
        plt.axis('off')


def extract_data():
    num_classes = len(breed_list)
    print("{} breeds".format(num_classes))

    n_total_images = 0
    for breed in breed_list:
        n_total_images += len(get_absolute_path(os.listdir("input/stanford-dogs-dataset/images/Images/{}".format(breed))))
    print("{} images".format(n_total_images))

    label_maps = {}
    label_maps_rev = {}
    for i, v in enumerate(breed_list):
        label_maps.update({v: i})
        label_maps_rev.update({i: v})

    print(breed_list[2])
    show_dir_images(breed_list[0], 16)

    if not os.path.exists(get_absolute_path('data')):
        os.mkdir(get_absolute_path('data'))
    for breed in breed_list:
        if not os.path.exists(get_absolute_path('data/' + breed)):
            os.mkdir(get_absolute_path('data/' + breed))
    print('Created {} folders to store cropped images of the different breeds.'.format(
        len(os.listdir(get_absolute_path('data')))))

    for breed in os.listdir(get_absolute_path('data')):
        for file in os.listdir(get_absolute_path('input/stanford-dogs-dataset/annotations/Annotation/{}'.format(breed))):
            img = Image.open(get_absolute_path('input/stanford-dogs-dataset/images/Images/{}/{}.jpg'.format(breed, file)))
            tree = ET.parse(get_absolute_path('input/stanford-dogs-dataset/annotations/Annotation/{}/{}'.format(breed, file)))
            xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
            xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
            ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
            ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img.save(get_absolute_path('data/' + breed + '/' + file + '.jpg'))
