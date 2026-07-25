from pathlib import Path

import cv2
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F

from utils.manager_shapefile import label_load, mask_load


class Sample(dict[str, torch.Tensor]):
    def mask_bbox(
        self,
        padding: int = 0,
    ) -> tuple[int, int, int, int]:
        """
        Возвращает bbox ненулевой области маски.

        Parameters
        ----------
        padding:
            Дополнительный отступ вокруг маски в пикселях.
            Отступ ограничивается границами изображения.

        Returns
        -------
        bbox:
            Кортеж ``(left, top, width, height)``.

            left:
                Минимальная координата по столбцам.

            top:
                Минимальная координата по строкам.

            width:
                Ширина области обрезки.

            height:
                Высота области обрезки.
        """
        if padding < 0:
            raise ValueError(
                f"padding не может быть отрицательным: {padding}"
            )

        mask = self["mask"]

        if mask.ndim == 2:
            binary_mask = mask != 0

        elif mask.ndim == 3 and mask.shape[0] == 1:
            binary_mask = mask[0] != 0

        elif mask.ndim == 3 and mask.shape[-1] == 1:
            binary_mask = mask[..., 0] != 0

        else:
            raise ValueError(
                "Ожидалась маска формы (H, W), (1, H, W) "
                f"или (H, W, 1), получена {tuple(mask.shape)}"
            )

        rows = torch.any(binary_mask, dim=1)
        columns = torch.any(binary_mask, dim=0)

        if not torch.any(rows):
            raise ValueError("Маска не содержит ненулевых пикселей")

        row_indices = torch.where(rows)[0]
        column_indices = torch.where(columns)[0]

        top = int(row_indices[0].item())
        bottom = int(row_indices[-1].item())

        left = int(column_indices[0].item())
        right = int(column_indices[-1].item())

        image_height, image_width = binary_mask.shape

        top = max(0, top - padding)
        bottom = min(image_height - 1, bottom + padding)

        left = max(0, left - padding)
        right = min(image_width - 1, right + padding)

        # bottom и right входят в bbox, поэтому нужен +1.
        width = right - left + 1
        height = bottom - top + 1

        return left, top, width, height

    def crop_to_mask(
        self,
        padding: int = 0,
    ) -> "Sample":
        """
        Обрезает весь сэмпл по bbox основной маски.

        Один и тот же прямоугольник применяется к:
            - image;
            - mask;
            - label.

        Исходный объект не изменяется.

        Parameters
        ----------
        padding:
            Дополнительный отступ вокруг bbox маски в пикселях.

        Returns
        -------
        cropped_sample:
            Новый обрезанный ``Sample``.
        """
        left, top, width, height = self.mask_bbox(
            padding=padding,
        )

        expected_size = self["mask"].shape[-2:]

        cropped: dict[str, torch.Tensor] = {}

        for name, tensor in self.items():
            if tensor.shape[-2:] != expected_size:
                raise ValueError(
                    f"Размер поля {name!r} не совпадает с маской: "
                    f"{tuple(tensor.shape[-2:])} != "
                    f"{tuple(expected_size)}"
                )

            cropped[name] = F.crop(
                tensor,
                top=top,
                left=left,
                height=height,
                width=width,
            )

        return Sample(cropped)

    def to_numpy(self) -> "Sample":
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