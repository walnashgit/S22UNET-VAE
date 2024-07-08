import math
import random

import torch
from matplotlib import pyplot as plt
from dataset.dataloader_util import denormalize


def validate_vae(model, dataset, num_images=25, is_cifar=False):
    input_images = []
    rand_label = []
    pred_images = []
    model.eval()
    dataiter = iter(dataset)
    for _ in range(num_images):
        image, label = next(dataiter)
        input_images.append(image)
        random_label_idx = random.randint(0,9)
        if random_label_idx == label:
            random_label_idx += random.choice([-1, 1])
        rand_label.append(random_label_idx)

        input = image.unsqueeze(0),  torch.tensor(random_label_idx).unsqueeze(0)
        x_hat = model(input)
        pred_images.append(x_hat.squeeze(0))

    plot_images(input_images, pred_images, rand_label, dataset, is_cifar=is_cifar)


def plot_images(input_images, pred_images, rand_labels, dataset, cols=6, is_cifar=False):
    rows = math.ceil(len(input_images * 2)/cols)
    plt.figure(figsize=(6, 12))
    c = 1
    r = 1
    idx = 1
    while r <= rows:
        while c <= cols or idx <= len(input_images):
            # plot input image
            plt.subplot(rows, cols, c)
            plt.tight_layout()
            img = denormalize(input_images[idx - 1].cpu(), is_cifar)
            plt.imshow(img.permute(1, 2, 0), aspect='auto')
            plt.title('-Input image-', fontsize=8)
            plt.xticks([])
            plt.yticks([])

            # plot output image
            lbl = rand_labels[idx - 1]
            plt.subplot(rows, cols, c)
            plt.tight_layout()
            img = denormalize(pred_images[idx - 1].detach().cpu(), is_cifar)
            plt.imshow(img.permute(1, 2, 0), aspect='auto')
            if is_cifar:
                lbl = dataset.classes[lbl]
            plt.title('Input label: ' + str(lbl), fontsize=8)
            plt.xticks([])
            plt.yticks([])
            c += 1
            idx += 1
    r += 1

    plt.show()









