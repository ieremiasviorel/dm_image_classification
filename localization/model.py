import torch
from torch import nn
from torchvision import models


class LocalizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.head = nn.Sequential(
            nn.BatchNorm1d(2048 * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.25),
            nn.Linear(in_features=2048 * 2, out_features=1024, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=4, bias=True),
        )

    def forward(self, x):
        x = self.body(x)
        x1 = nn.AdaptiveAvgPool2d(1)(x)
        x2 = nn.AdaptiveMaxPool2d(1)(x)
        x = torch.cat([x1, x2], 1)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x
