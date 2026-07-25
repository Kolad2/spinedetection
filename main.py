from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from utils.tensor_inertia import vertical_deviation
from utils.sample import Sample

from utils.sample_transform import get_random_affine_sample, calculate_transform_limits


def main() -> None:
    folder_dataset = Path(r"D:\Data\humanspine\dataset")

    sample_ids = [3, 76, 7]

    samples_paths = [
        sample_paths_from_folder(folder_dataset / str(sample_id))
        for sample_id in sample_ids
    ]

    samples = [
        Sample.load_sample(paths)
        for paths in samples_paths
    ]

    transformed_samples: list[Sample] = []
    samples = [sample.crop_to_mask() for sample in samples]

    for sample in samples:
        scale_range, angle_range = calculate_transform_limits(
            sample,
            scale_deviation=(0.7, 1.3),
            angle_deviation=(-10.0, 10.0),
        )

        transformed_sample = get_random_affine_sample(
            sample,
            scale_range=scale_range,
            angle_range=angle_range,
            expand=True,
        )
        transformed_samples.append(transformed_sample.to_numpy())

    original_samples = [
        Sample(dict(sample)).to_numpy()
        for sample in samples
    ]

    images = [
        to_image(sample)
        for sample in original_samples
    ]

    transformed_images = [
        to_image(sample)
        for sample in transformed_samples
    ]

    fig, axes = plt.subplots(
        2,
        len(sample_ids),
        figsize=(5, 7),
        squeeze=False,
    )

    for index, sample_id in enumerate(sample_ids):
        axes[0, index].imshow(images[index])
        axes[0, index].set_title(
            f"Sample {sample_id}: original"
        )
        axes[0, index].set_aspect("equal")
        axes[0, index].axis("off")

        axes[1, index].imshow(transformed_images[index])
        axes[1, index].set_title(
            f"Sample {sample_id}: transformed"
        )
        axes[1, index].set_aspect("equal")
        axes[1, index].axis("off")

    plt.tight_layout()
    plt.show()


def sample_paths_from_folder(
    folder_sample: Path,
) -> dict[str, Path]:
    return {
        "image": folder_sample / f"{folder_sample.name}_3.jpeg",
        "mask": folder_sample / f"{folder_sample.name}_3_vectormask",
        "label": folder_sample / f"{folder_sample.name}_3_vector",
    }


def to_image(sample: dict) -> np.ndarray:
    image = sample["image"].copy()

    image[sample["mask"] == 0] = 0
    image[sample["label"] == 255] = 255

    return image


def draw_deviation_axis(
    ax,
    angle_deg: float,
    w: int,
    h: int,
    color: str = "red",
    linewidth: float = 2.0,
) -> None:
    theta = np.deg2rad(angle_deg)

    # dx — вниз, dy — вправо.
    dx = np.cos(theta)
    dy = np.sin(theta)

    cx = w / 2
    cy = h / 2

    length = 0.45 * min(w, h)

    x1 = cx - dy * length
    y1 = cy - dx * length

    x2 = cx + dy * length
    y2 = cy + dx * length

    ax.plot(
        [x1, x2],
        [y1, y2],
        color=color,
        linewidth=linewidth,
    )

    ax.scatter(
        [cx],
        [cy],
        s=20,
        color=color,
    )


if __name__ == "__main__":
    main()