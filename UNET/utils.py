import matplotlib.pyplot as plt
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'


def plot_prediction_sample(input, target, pred):
    i = 0
    if input is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(input.cpu().permute(1, 2, 0))

    if target is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(target.cpu().permute(1, 2, 0))

    if pred is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(pred.cpu().permute(1, 2, 0))