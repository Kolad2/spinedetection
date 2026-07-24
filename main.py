from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from utils.manager_shapefile import label_load, mask_load
from utils.sample import Sample

def main() -> None:
    folder_dataset = Path(r"D:\Data\humanspine\dataset")
    folder_sample = folder_dataset / "3"

    path_sample = {
        "image": folder_sample / f"{folder_sample.name}_3.jpeg",
        "mask": folder_sample / f"{folder_sample.name}_3_vectormask",
        "label": folder_sample / f"{folder_sample.name}_3_vector",
    }

    sample = Sample.load_sample(path_sample)

    rotate = transforms.RandomRotation(
        # Строго +15 градусов.
        degrees=(15, 15),

        # Для изображения применяется bilinear.
        # Для tv_tensors.Mask torchvision использует nearest.
        interpolation=InterpolationMode.BILINEAR,

        fill={
            tv_tensors.Image: 0,
            tv_tensors.Mask: 0,
        },

        # Холст увеличивается, чтобы вместить всё изображение.
        expand=True,
    )

    # Один и тот же поворот применяется ко всему sample.
    sample = rotate(sample)
    sample = sample.to_numpy()

    image_rotated = sample["image"]  # H x W x C
    mask_rotated = sample["mask"]    # H x W
    label_rotated = sample["label"]  # H x W

    # Удаляем фон по основной маске.
    image_masked = image_rotated.copy()
    image_masked[mask_rotated == 0] = 0

    # Создаём изображение с нанесённым лейблом.
    image_with_label = image_masked.copy()

    # Красным отмечаем пиксели ломаной линии.
    image_with_label[label_rotated != 0] = (255, 0, 0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
    )

    axes[0].imshow(image_with_label)
    axes[0].set_title("Image + label: rotation +15°")
    axes[0].axis("off")

    axes[1].imshow(
        mask_rotated,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1].set_title("Mask: rotation +15°")
    axes[1].axis("off")

    axes[2].imshow(
        label_rotated,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[2].set_title("Label: rotation +15°")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()