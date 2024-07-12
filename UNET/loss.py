import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target):
    smooth = 1e-5

    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


def bce_loss(pred, target):
    #y = pred.view(pred.size(0), -1)
    y = F.softmax(pred, dim=1)

    #y = y.view(y.size(0), -1)
    #target = target.view(target.size(0), -1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y, target)

    return loss