import sys
sys.path.insert(0, r'D:\Python_packages')
import torch.nn as nn
from torchvision import models

def build_model(num_classes=7, freeze_backbone=True):
    model = models.efficientnet_b3(weights='DEFAULT')

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')

    return model

def unfreeze_backbone(model, unfreeze_last_n_blocks=3):
    for param in model.classifier.parameters():
        param.requires_grad = True

    blocks = list(model.features.children())
    for block in blocks[-unfreeze_last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'After unfreezing: {trainable:,} trainable parameters')

    return model
