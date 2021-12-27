import os
import numpy as np
from PIL import Image as Image
import torch
import random as rd
# from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from albumentations import HorizontalFlip, RandomCrop, CenterCrop, ShiftScaleRotate
from albumentations.core.composition import Compose
# from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



def get_train_dataset(path, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = Compose(
            [
                RandomCrop(height=256, width=256),
                HorizontalFlip(p=0.5),
            ],
            additional_targets={'image2': 'image'}
        )
    return GoProDataset(image_dir, transform=transform)


def get_test_dataset(path):
    image_dir = os.path.join(path, 'test')

    return GoProDataset(image_dir)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def make_dataset_several(dirs):
    images = []
    for dir in dirs:
        if not os.path.isdir(dir):
            continue
        # assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    # print(path)
                    images.append(path)

    return images

def get_A_paths(dataset_path):
    subfolders = os.listdir(os.path.join(dataset_path, 'blur'))
    dirs_A = [os.path.join(dataset_path, 'blur', subfolder) for subfolder in subfolders]

    A_paths = make_dataset_several(dirs_A)

    return A_paths


class GoProDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.SEED_MAX = 2147483647
        self.A_paths = get_A_paths(dataset_path=image_dir)
        self.B_paths = self.get_GT_sharp_paths()

        self.transform = transform

    def __len__(self):
        return len(self.A_paths)

    def get_GT_sharp_paths(self):
        def change_subpath(path, what_to_change, change_to):
            p = Path(path)
            index = p.parts.index(what_to_change)
            new_path = (Path.cwd().joinpath(*p.parts[:index])).joinpath(Path(change_to),
                                                                        *p.parts[index + 1:])
            return new_path

        B_paths = [str(change_subpath(x, 'blur', 'sharp')) for x in self.A_paths]
        return B_paths

    def __getitem__(self, idx):
        image, label = Image.open(self.A_paths[idx]), Image.open(self.B_paths[idx])
        # image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        # label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        # As we are given transform for one image, we should apply seed
        image, label = self.transform(image=image, image2=label)
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError