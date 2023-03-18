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
        
        # add the start token.
        target = torch.tensor([1] + target, dtype=torch.long)

        image = Image.open(os.path.join(self.root, path)).convert('RGBA').getchannel("A")
        image = np.array(image) / 255
        image = torch.tensor(image, dtype=torch.float32)
        n, m = image.shape

        # Random Resize
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        max_scale = min(max_h / n, max_w / m)

        ratio = np.random.uniform(0.6, max_scale)
        n, m = int(n*ratio), int(m*ratio)

        image = image.unsqueeze(0)

        a, b = np.random.randint(0, max_w-m+1), np.random.randint(0, max_h-n+1)
        transform = transforms.Compose([
            transforms.Resize( (n, m) ),
            transforms.Pad( (a, b, max_w-m-a, max_h-n-b), fill=0, padding_mode='constant'),
        ])
        image = transform(image)
        
        return image, target

    def __len__(self):
        return self.meta.shape[0]
