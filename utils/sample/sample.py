from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

from utils.manager_shapefile import label_load, mask_load


class Sample(dict[str, torch.Tensor]):
    def to_numpy(self) -> Sample:
        for name, tensor in self.items():
            array = tensor.detach().cpu().numpy()

            # Изображение: CHW -> HWC.
            if name == "image":
                if array.ndim != 3:
                    raise ValueError(
                        f"Ожидалось изображение CHW, получена форма: "
                        f"{array.shape}"
                    )
                array = np.transpose(
                    array,
                    (1, 2, 0),
                )
            self[name] = array
        return self

    @classmethod
    def load_sample(
        cls,
        path_sample: dict[str, Path],
        thickness: int = 1,
    ) -> "Sample":
        image = cv2.imread(
            str(path_sample["image"])
        )

        if image is None:
            raise FileNotFoundError(
                "Не удалось загрузить изображение: "
                f"{path_sample['image']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = mask_load(
            path_sample["mask"],
            image.shape,
        )

        label = label_load(
            path_sample["label"],
            image.shape,
            thickness=thickness,
        )

        # На случай формы H x W x 1.
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if label.ndim == 3 and label.shape[-1] == 1:
            label = label[..., 0]

        return cls({
            "image": transforms.ToImage()(image),

            "mask": tv_tensors.Mask(
                torch.from_numpy(mask)
            ),

            "label": tv_tensors.Mask(
                torch.from_numpy(label)
            ),
        })