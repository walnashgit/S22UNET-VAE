import pytorch_lightning as pl
import torch

from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule
from vae import VAE
from dataset.dataloader_util import *
from utils import validate_vae

# datamodule = CIFAR10DataModule('.')
# datamodule = MNISTDataModule('.')

cuda = torch.cuda.is_available()
dataloader_args = dict(shuffle = True, batch_size = 32, num_workers = 2, pin_memory = True) if cuda else dict(shuffle = True, batch_size = 256, pin_memory = True)
train, test = get_MNIST_dataloader(**dataloader_args)
# train, test = get_CIFAR10_dataloader(**dataloader_args)


vae = VAE()

# tr = pl.Trainer(max_epochs = 1, precision = 16, accelerator='mps')
# tr.fit(vae, train)
# tr.fit(vae, datamodule=datamodule)

train_set, test_set = get_mnist_dataset()
# train_set, test_set = get_cifar10_dataset()
validate_vae(vae, test_set, is_cifar=False)


