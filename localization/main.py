import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from definitions import IMAGES_DIR, ANNOTATIONS_DIR
from localization.dataset import LocalizationDataset
from localization.model import LocalizationModel
from localization.train import IoU

EPOCH = 10
BS = 64
LR = 1e-1

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

valid_ds = LocalizationDataset(all_img_name[int(len(all_img_name) * 0.8):int(len(all_img_name) * 0.9)], transform=tsfm)
valid_dl = DataLoader(valid_ds, batch_size=1)

test_ds = LocalizationDataset(all_img_name[int(len(all_img_name) * 0.9):], is_test=True, transform=test_tsfm)
test_dl = DataLoader(test_ds, batch_size=1)

train_ds.show_box()

model = LocalizationModel()
# model.cuda()

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
        # imgs = imgs.cuda()
        # boxes = boxes.cuda()

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
    print('epoch:', epoch, '/', EPOCH, '\ttrain_loss:', np.mean(train_loss), '\tvalid_loss:', np.mean(valid_loss),
          '\tIoU:', np.mean(valid_IoU))
