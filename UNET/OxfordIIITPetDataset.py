from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
import numpy as np

import torch

dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)


class SegmentOxfordIIITPetDataset(Dataset):
    def __init__(self, root='../data', download=True, train=True, input_transform=None, mask_transform=None):
        if train:
            split = 'trainval'
        else:
            split = 'test'

        self.ds = torchvision.datasets.OxfordIIITPet(root=root,
                                                     target_types='segmentation',
                                                     download=download,
                                                     split=split)

        self.input_transform = input_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        data, seg = self.ds[idx]

        if self.input_transform:
            data = self.input_transform(data)

        if self.mask_transform:
            seg = self.mask_transform(seg)

            seg = torch.Tensor(np.array(seg))

            seg1 = (seg == 1) * 1.0
            seg2 = (seg == 2) * 1.0
            seg3 = (seg == 3) * 1.0

            seg = torch.stack([seg1, seg2, seg3], dim=0)

        return data, seg

    def __len__(self):
        return len(self.ds)


def get_dataset():
    image_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )

    mask_transform = T.Compose(
        [
            T.Resize((224, 224))
        ]
    )

    train_data = SegmentOxfordIIITPetDataset(train=True, download=True, input_transform=image_transform,
                                             mask_transform=mask_transform)
    test_data = SegmentOxfordIIITPetDataset(train=False, download=True, input_transform=image_transform,
                                            mask_transform=mask_transform)

    return train_data, test_data


def get_dataloader(**kwargs):
    train_data, test_data = get_dataset()
    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)