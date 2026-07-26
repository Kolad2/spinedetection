import cv2
import torch
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, lst_path, path_root=None):
        self.path_root = path_root if path_root is not None else lst_path.parent
        self.items = []

        with open(lst_path, "r") as f:
            for line in f:
                image, mask = line.strip().split()
                self.items.append(
                    (image, mask)
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label_path = self.items[idx]
        image_path = str(self.path_root / image_path)
        label_path = str(self.path_root / label_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = cv2.imread(
            label_path,
            cv2.IMREAD_GRAYSCALE
        )

        # 0/255 -> 0/1
        label = label.astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(
            image.transpose(2,0,1)
        )

        label = torch.from_numpy(
            label[None,:,:]
        )

        return image, label