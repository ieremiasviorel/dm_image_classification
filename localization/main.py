import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from definitions import IMAGES_DIR, ANNOTATIONS_DIR
from definitions import absolute_path
from localization.dataset import LocalizationDataset
from localization.model import LocalizationModel
from localization.train import IoU

EPOCH = 10
BS = 64
LR = 1e-1


def main_flow():
    with open(absolute_path('input/annotations/Annotation/n02113799-standard_poodle/n02113799_489')) as f:
        reader = f.read()

    img = Image.open(absolute_path('input/images/Images/n02113799-standard_poodle/n02113799_489.jpg'))

    xmin = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)[0])
    xmax = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)[0])
    ymin = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)[0])
    ymax = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)[0])

    origin_img = img.copy()
    draw = ImageDraw.Draw(origin_img)
    draw.rectangle(xy=[(xmin, ymin), (xmax, ymax)])
    print(origin_img)

    all_img_folder = os.listdir(IMAGES_DIR)
    all_annotation_folder = os.listdir(ANNOTATIONS_DIR)

    all_img_name = []
    for img_folder in all_img_folder:
        img_folder_path = '{}/{}'.format(IMAGES_DIR, img_folder)
        all_img_name += list(map(lambda x: img_folder + '/' + x, os.listdir(img_folder_path)))

    all_annotation_name = []
    for annotation_folder in all_annotation_folder:
        annotation_folder_path = '{}/{}'.format(ANNOTATIONS_DIR, annotation_folder)
        all_annotation_name += list(map(lambda x: annotation_folder + '/' + x, os.listdir(annotation_folder_path)))

    print(len(all_img_name), all_img_name[0])
    print(len(all_annotation_folder), all_annotation_name[0])

    tsfm = transforms.Compose([
        transforms.Resize([332, 332]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_tsfm = transforms.Compose([
        transforms.Resize([332, 332]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LocalizationDataset(all_img_name[:int(len(all_img_name) * 0.8)], transform=tsfm)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)

    valid_ds = LocalizationDataset(all_img_name[int(len(all_img_name) * 0.8):int(len(all_img_name) * 0.9)],
                                   transform=tsfm)
    valid_dl = DataLoader(valid_ds, batch_size=1)

    test_ds = LocalizationDataset(all_img_name[int(len(all_img_name) * 0.9):], is_test=True, transform=test_tsfm)
    test_dl = DataLoader(test_ds, batch_size=1)

    train_ds.show_box()

    model = LocalizationModel()
    model.cuda()

    for layer in model.body.parameters():
        layer.requires_grad = False

    optm = torch.optim.Adam(model.head.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCH + 1):
        if epoch == 3:
            optm = torch.optim.Adam(model.head.parameters(), lr=LR / 2)
        elif epoch == 5:
            optm = torch.optim.Adam(model.head.parameters(), lr=LR / 4)
        elif epoch == 7:
            optm = torch.optim.Adam(model.head.parameters(), lr=LR / 8)
        elif epoch == 9:
            optm = torch.optim.Adam(model.head.parameters(), lr=LR / 10)

        train_loss = []
        train_IoU = []

        for step, data in enumerate(train_dl):
            imgs, boxes = data
            imgs = imgs.cuda()
            boxes = boxes.cuda()

            pred = model(imgs)
            loss = nn.L1Loss()(pred, boxes.squeeze())
            train_loss.append(loss.item())
            IOU = IoU(pred, boxes)
            train_IoU.append(IOU)

            optm.zero_grad()
            loss.backward()
            optm.step()
            if step % 10 == 0:
                print('step: ', step, '/', len(train_dl), '\tloss:', loss.item(), '\tIoU:', float(IOU.mean()))

        model.eval()
        valid_loss = []
        valid_IoU = []

        for step, data in enumerate(tqdm(valid_dl)):
            imgs, boxes = data
            imgs = imgs.cuda()
            boxes = boxes.cuda()

            pred = model(imgs)
            loss = nn.L1Loss()(pred, boxes.squeeze())
            valid_loss.append(loss.item())
            IOU = IoU(pred, boxes)
            valid_IoU.append(IOU.item())
        print('epoch:', epoch, '/', EPOCH, '\ttrain_loss:', np.mean(train_loss), '\tvalid'
                                                                                 '_loss:', np.mean(valid_loss),
              '\tIoU:', np.mean(valid_IoU))

    torch.save(model.state_dict(), absolute_path('localization_01.pt'))
    model.eval()

    draw_img = []
    for step, img in enumerate(tqdm(test_dl)):
        img = img.cuda()
        pred = model(img)

        origin_img = transforms.ToPILImage()(img.cpu().squeeze())
        draw = ImageDraw.Draw(origin_img)
        xmin, ymin, xmax, ymax = tuple(pred.squeeze().tolist())
        draw.rectangle(xy=[(int(xmin), int(ymin)), (int(xmax), int(ymax))])
        draw_img.append(origin_img)


main_flow()
