from loss import bce_loss
from torchsummary import summary
from utils import device
import pytorch_lightning as pl
from unet_model import UNet


unet_config=dict(
    image_size=224,
    batch_size=32,
    num_epochs=15
)

def init(
        train_dataloader,
        val_dataloader,
        cfg=None,
        in_channels=3,
        out_channels=1,
        show_summary=False,
        max_lr=None,
        loss_fn=bce_loss,
        upsample='transpose_conv',
        downsample='maxpool',
        accelerator=None
):
    model = UNet(in_channels, out_channels, max_lr, loss_fn, upsample, downsample)

    # if show_summary:
    #     summary(model.to(device), input_size=(in_channels, cfg['image_size'], cfg['image_size']))
        # summary(model.to('mps'), input_size=(in_channels, cfg['image_size'], cfg['image_size']))

    trainer_args = dict(
        precision='16',
        max_epochs=cfg['num_epochs']
    )

    if accelerator:
        # trainer_args['precision'] = '16-mixed'
        trainer_args['accelerator'] = accelerator

    trainer = pl.Trainer(
        **trainer_args
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return model
