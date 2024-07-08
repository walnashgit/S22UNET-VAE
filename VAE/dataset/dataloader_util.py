import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset

import torch


def get_mnist_stddev_mean():
    return (0.1307,), (0.3081,)


def get_cifar_stddev_mean():
    return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class MultiChannelMNIST(Dataset):
    def __init__(self, root='../data', download=True, train=True, transform=None):

        self.ds = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)

        self.transform = transform

    def __getitem__(self, idx):
        data, label = self.ds[idx]

        data = torch.stack([data.squeeze(0), data.squeeze(0), data.squeeze(0)], dim=0)

        return data, label

    def __len__(self):
        return len(self.ds)


def get_cifar10_dataset():
    dataset_mean, dataset_std = get_cifar_stddev_mean()

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )

    train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=image_transform)
    test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=image_transform)

    return train_data, test_data


def get_CIFAR10_dataloader(**kwargs):
    train_data, test_data = get_cifar10_dataset()
    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)


def get_mnist_dataset():
    dataset_mean, dataset_std = get_mnist_stddev_mean()
    img_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    train_data = MultiChannelMNIST('../data', train=True, download=True, transform=img_transforms)
    test_data = MultiChannelMNIST('../data', train=False, download=True, transform=img_transforms)

    return train_data, test_data

def get_MNIST_dataloader(**kwargs):
    train_data, test_data = get_mnist_dataset()
    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)


def denormalize(img, is_cifar=False):
    if is_cifar:
        std_dev, mean = get_cifar_stddev_mean()
        for i in range(img.shape[0]):
            img[i] = (img[i] * std_dev[i]) + mean[i]
    # else:
    #     std_dev, mean = get_mnist_stddev_mean()
    #     img = img * std_dev + mean
    #     img = img * 255

    return img

