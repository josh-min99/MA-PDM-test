import torch
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory, nrow=8, padding=1):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory, nrow=nrow, padding=padding)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth')


def load_checkpoint(path, device):
    # PyTorch>=2.6 defaults weights_only=True, which rejects the argparse.Namespace
    # (config/params) stored in our own checkpoints. These are self-produced/trusted.
    if device is None:
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path, map_location=device, weights_only=False)
