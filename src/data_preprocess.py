import torch
from torch import Tensor
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt
import os
import os.path as op
import numpy as np
from typing import List, Dict, Tuple

DATAPATH = path = '../data/imagenette2-320'

# Define CONSTANTS: LABELS: dict mapping folder names to labels; IDS: dict mapping labels to integer id
IMAGENETTE_LABELS = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)

IMAGENETTE_LABEL_IDS = {
    'tench': 0,
    'English springer': 1,
    'cassette player': 2,
    'chain saw': 3,
    'church': 4,
    'French horn': 5,
    'garbage truck': 6,
    'gas pump': 7,
    'golf ball': 8,
    'parachute': 9,
}

# Training set mean and stdev; Found these manually
IMAGENETTE_MEAN = np.array([0.46254329, 0.45792598, 0.42990307])*255
IMAGENETTE_STDEV = np.array([0.24124826, 0.23532296, 0.24335882])*255

# Tensors
IMAGENETTE_MEAN2 = torch.as_tensor(IMAGENETTE_MEAN, dtype=torch.float).view((1, 3, 1, 1))
IMAGENETTE_STDEV2 = torch.as_tensor(IMAGENETTE_STDEV, dtype=torch.float).view((1, 3, 1, 1))


class ToTensor:
    """Convert np.ndarray in sample to Tensors.
    Numpy array is HxWxC, but we want tensors of CxHxW
    """
    def __call__(self, image: np.ndarray) -> Tensor:

        image = image.transpose((2, 0, 1))
        return torch.as_tensor(image.copy(), dtype=torch.float)


class Imagenette(Dataset):
    """Primary class that handles imagenette dataset
    Guided by:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Attributes
    ----------
    data_dir:
        Directory containing folders whose names are class labels, and whose contents contain images of corresponding
        class label.
    file_paths: List
        List of full paths to every image in specified (train or validation) dataset.
    label_dict: Dict
        Dictionary mapping folder names (strings) to class labels (strings).
    label_ids: Dict
        Dictionary mapping class labels (strings) to class IDs (integers).
    transform: torchvision.transform
    train: Bool, optional
        Whether or not this dataset object is training or validation set.
    requires_grad: Bool, optional
        Whether or not input has grad = True.
    """

    def __init__(self, root_dir: str, train: bool = True, transform: torchvision.transforms = None,
                 requires_grad: bool = False, device: torch.device = torch.device('cpu')):
        self.data_dir = op.join(root_dir, 'train' if train else 'val')
        self.file_paths = []

        for folder in os.listdir(self.data_dir):
            files = os.listdir(op.join(self.data_dir, folder))
            self.file_paths.extend([op.join(folder, f) for f in files])

        self.label_dict = IMAGENETTE_LABELS.copy()
        self.label_ids = IMAGENETTE_LABEL_IDS.copy()

        self.transform = transform
        self.train = train
        self.requires_grad = requires_grad
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.file_paths[idx]
        label_key = image_path.split('/')[0]
        label = self.label_dict[label_key]
        label_id = self.label_ids[label]
        image = io.imread(op.join(self.data_dir, image_path))

        if len(image.shape) < 3:
            image = np.stack([image]*3, axis=-1)
        sample = {'image': image,
                  'label': torch.as_tensor(label_id, device=self.device)}

        if self.transform:
            sample['image'] = self.transform(sample['image']).requires_grad_(self.requires_grad)

        return sample


class ImagenetteDataLoader(DataLoader):
    """Wrapper that creates DataLoader with Imagenette Dataset """
    def __init__(self, root_dir, train=True, crop_size=256, transform=None, requires_grad=False,  **kwargs):
        """
        Parameters
        ----------
        root_dir: str
            Root directory of data. See Imagenette() class docstring for details.
        train: Bool
            Whether or not we will be using training vs validation set.
        crop_size: Tuple or int
            For random crop. If integer, then it will be square crop
        transform: transforms.Transform or transforms.Compose
            Transform to apply to data. Note: we MUST first apply a ToTensor() transform to change the dtype and
            permute axes of the image to CxHxW.
        kwargs
            Keyword arguments for DataLoader initialization.
        """

        if transform is None:
            transform = transforms.Compose([ToTensor(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(crop_size)
                                           ])
        dataset = Imagenette(root_dir=root_dir, train=train, transform=transform, requires_grad=requires_grad)
        super().__init__(dataset, **kwargs)

    def show_batch(self, max_plot: int = 5):
        """Plots max_plot images from batch. Images are rescaled to [0,1]"""
        size = self.batch_size if self.batch_size <= max_plot else max_plot
        fig, ax = plt.subplots(1, size, figsize=(10, 10), sharey='all', sharex='all')
        ax[0].set(xticks=[], yticks=[])
        fig.tight_layout()
        for batch in self:

            for i, im in enumerate(batch['image'][:max_plot]):
                im = np.array(im.permute((1, 2, 0)).int())
                id = int(batch['label'][i])
                label = next(key for key, value in self.dataset.label_ids.items() if value == id)
                max_val = im.max(axis=(0,1)).reshape((1,1,3))
                min_val = im.min(axis=(0,1)).reshape((1,1,3))

                im = (im - min_val)/(max_val-min_val)
                ax[i].imshow(im)
                ax[i].set(title=label)
            break


def get_train_validation_data(root_dir: str,
                              batch_size: int = 64,
                              crop_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """Helper function to create training and validation data
    Normalization mean and standard deviations were found manually by averaging channel mean and stdev aross all
    images in training set.

    Parameters
    ----------
    root_dir: str
        Root directory of data.
    batch_size: int
        Size of batch for both training and validation sets.
    crop_size: int
        Height and width of square input images in training and validation set.

    Returns
    -------
    train_loader: DataLoader
        Imagenette Dataloader for training data
    valid_loader: DataLoader
        Imagenette Dataloader for validation data
    """

    transform = transforms.Compose([ToTensor(),
                                    transforms.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STDEV),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(crop_size)
                                    ])

    train_loader = ImagenetteDataLoader(root_dir=root_dir, train=True, batch_size=batch_size, shuffle=True,
                                        num_workers=0, transform=transform, requires_grad=True)

    valid_loader = ImagenetteDataLoader(root_dir=root_dir, train=False, batch_size=batch_size, shuffle=True,
                                        num_workers=0, transform=transform, requires_grad=True)

    return train_loader, valid_loader


def make_synthesis_test_images(crop_size: int = 256, seed: int = 0):
    """Deterministically crop images for eigendistortion synthesis"""

    torch.manual_seed(seed)
    transform = transforms.Compose([ToTensor(),
                                    transforms.RandomCrop(crop_size),
                                    lambda x: x.permute((1, 2, 0)).numpy().astype(np.uint8)  # convert back to numpy
                                    ])

    save_dir = op.join('../data/to_synthesize')

    # hard code test images (folder, filename, newfilename)
    images = [("n01440764", "n01440764_4562.JPEG", "fish"),
              ("n02102040", "n02102040_762.JPEG", "dog"),
              ("n03028079", "n03028079_20210.JPEG", "church"),
              ("n03394916", "n03394916_56022.JPEG", "horn"),
              ("n03417042", "n03417042_4470.JPEG", "truck"),
              ]

    for folder, file, newfile in images:
        print(folder, file, newfile)
        img = plt.imread(op.join(DATAPATH, 'val', folder, file))
        cropped_img = transform(img)
        print(cropped_img.shape)
        filename = op.join(save_dir)

        # with open( 'w+') as f:
        # np.save(file=f"{filename}/{newfile}.npy", arr=cropped_img)
        plt.imsave(fname=f"{filename}/{newfile}.jpeg", arr=cropped_img)


if __name__ == '__main__':

    # test if Dataset works
    # trans = transforms.Compose([ToTensor(),
    #                             transforms.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STDEV),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.RandomCrop(256)
    #                             ])

    # dataset = Imagenette(DATAPATH, transform=trans)

    # plt.imshow(dataset[0]['image'].permute((1, 2, 0)).int())
    # plt.show()

    # test if DataLoader works
    # loader = ImagenetteDataLoader(root_dir='../data/imagenette2-320', train=True,
    #                               batch_size=64, shuffle=True, num_workers=0, transform=trans)
    # for i, im in enumerate(loader):
    #     print(im['image'].shape)
    #     print(im['label'])
    #     break

    make_synthesis_test_images()

    # img = np.load(op.join('../data/to_synthesize/fish.npy'), allow_pickle=True)
    # print(img.shape)
