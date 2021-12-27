import os
import numpy as np
from PIL import Image as Image
import torch
import random as rd
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def get_train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = Compose(
            [
                RandomCrop(256),
                RandomHorizontalFlip(),
                ToTensor()
            ]
        )
    dataloader = DataLoader(
        GoProDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        GoProDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def get_valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        GoProDataset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class GoProDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.SEED_MAX = 2147483647

        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        # As we are given transform for one image, we should apply seed
        seed = np.random.randint(self.SEED_MAX)
        if self.transform:
            rd.seed(seed)
            image = self.transform(image)

            rd.seed(seed)
            label = self.transform(label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError