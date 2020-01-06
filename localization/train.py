import torch
from torch import nn


def loss_fn(preds, targs, class_idxs):
    return nn.L1Loss()(preds, targs.squeeze())


def IoU(preds, targs):
    return intersection(preds, targs.squeeze()) / union(preds, targs.squeeze())


def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    if len(targs.shape) == 1:
        targs = targs.reshape(1, 4)
    #     print(preds.shape, targs.shape)
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def union(preds, targs):
    if len(targs.shape) == 1:
        targs = targs.reshape(1, 4)
    return area(preds) + area(targs) - intersection(preds, targs)
