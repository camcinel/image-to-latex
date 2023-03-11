import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import pandas as pd


class MyDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, meta_data, img_size, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            meta_data_path: path for the meta data.
            img_size: image size, a tuple (width, height).
            transform: image transformations.
        """
        self.root = root
        self.meta = meta_data
        # reindex the meta data.
        self.meta.index = range(self.meta.shape[0])
        self.img_size = img_size


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        path = self.meta['image'][index]

        target = self.meta['padded_seq'][index]
        target = torch.tensor(target, dtype=torch.long)

        # TODO: add normalization, change the padding for centering the image.
        image = Image.open(os.path.join(self.root, path)).convert('RGBA').getchannel("A")
        image = np.array(image) / 255
        print(image[0])
        image = torch.tensor(image, dtype=torch.float32)
        n, m = image.shape
        transform = transforms.Compose([
            transforms.Pad((0, 0, self.img_size[1] - m, self.img_size[0] - n), fill=0, padding_mode='constant'),
        ])
        image = transform(image)
        return image, target

    def __len__(self):
        return self.meta.shape[0]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    by padding the captions to make them of equal length.

    We can not use default collate_fn because variable length tensors can't be stacked vertically.
    We need to pad the captions to make them of equal length so that they can be stacked for creating a mini-batch.

    Read this for more information - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_ids = zip(*data)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, img_ids
