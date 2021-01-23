import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from ..data import IMAGENETTE_STDEV2, IMAGENETTE_MEAN2

""" Image tools for handling eigendistortions
"""

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def preprocess_image(image_dir, seed=0):
    """Prepares image for analysis. Subtracts training set mean, scales by training set stdev, and randomly crops a
    256 square."""
    torch.manual_seed(seed)
    img_np = plt.imread(image_dir).copy()
    img = torch.as_tensor(img_np, dtype=torch.float).to(DEVICE)
    img = img.permute((2, 0, 1)).unsqueeze(0)
    img = (img - IMAGENETTE_MEAN2)/IMAGENETTE_STDEV2
    cropper = transforms.RandomCrop(256)
    return cropper(img)


def vec_to_image(vec):
    """Reshapes eigendistortion 1D vector to torch image (b, c, h, w)"""
    return torch.reshape(vec, (1, 3, 256, 256))


def rescale(img):
    """Rescales an image to [0, 1]"""
    return (img - img.min())/(img.max() - img.min())


def unprocess(img):
    """Undoes the transformation done by preprocess_image
    Tensor image -> numpy image"""
    x = (img*IMAGENETTE_STDEV2) + IMAGENETTE_MEAN2
    x = x / 255.
    x = x.squeeze().permute((1, 2, 0))
    return x.numpy()


def clamp(x, min_val=None, max_val=None):
    """Clamps an image between min_val and max_val"""
    min_val = x.min() if min_val is None else min_val
    max_val = x.max() if max_val is None else max_val

    x[x < min_val] = min_val
    x[x > max_val] = max_val

    return x
