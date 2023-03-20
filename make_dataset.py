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
        
        # self.normalize = transforms.Compose([
        #                                      transforms.ToTensor(),
        #                                      transforms.Normalize(mean=5.96457, std=38.54074)
        #                                     ])
        

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        path = self.meta['image'][index]

        target = self.meta['padded_seq'][index]
        
        # add the start token.
        target = torch.tensor([1] + target, dtype=torch.long)

        # TODO: add normalization, change the padding for centering the image.
        image = Image.open(os.path.join(self.root, path)).convert('RGBA').getchannel("A")

        n, m = image.size
        resize_transform = transforms.Compose([
                                               transforms.Resize((n//2, m//2), interpolation=transforms.InterpolationMode.BILINEAR),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=5.96457, std=38.54074)
                                              ])

        image = resize_transform(image).squeeze(0)

        n1, m1 = image.shape
        
        a, b = (self.img_size[1] - m1) // 2, (self.img_size[0] - n1) // 2   # (width, height)
        pad_transform = transforms.Compose([
            transforms.Pad( (a, b, self.img_size[1]-m1-a, self.img_size[0]-n1-b), fill=0, padding_mode='constant'),
        ])
        image = pad_transform(image).unsqueeze(0)
        
        return image, target

    def __len__(self):
        return self.meta.shape[0]
