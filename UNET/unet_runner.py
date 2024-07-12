from main import init, unet_config
from OxfordIIITPetDataset import *
from loss import bce_loss, dice_loss




batch_size = 32
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}

train_dataloader, test_dataloader = get_dataloader()

print('##-- MP+Tr+BCELoss --##')
model = init(train_dataloader,
             test_dataloader,
             cfg=unet_config,
             in_channels=3,
             out_channels=3,
             show_summary=True,
             max_lr=10e-3,
             loss_fn=bce_loss,
             upsample='transpose_conv',
             downsample='maxpool',
             accelerator='cpu')