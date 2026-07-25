from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from utils.tensor_inertia import vertical_deviation
from utils.sample import Sample

def sample_process(
    sample: Sample,
    etalon_area: int = 3_000_000,
) -> tuple[Sample, float, float, float]:
    """
    Выравнивает главную ось маски по вертикали и приводит площадь
    маски к заданному эталонному значению.

    Returns
    -------
    processed_sample:
        Повёрнутый и масштабированный сэмпл.

    angle_before:
        Исходное знаковое отклонение от вертикали.

    angle_after:
        Отклонение после обработки.

    scale:
        Применённый линейный коэффициент масштаба.
    """

    angle_before = float(
        vertical_deviation(
            sample["mask"],
            normalize=True,
            degrees=True,
        ).detach().cpu().item()
    )

    # Если объект отклонён на angle_before,
    # для исправления поворачиваем на противоположный угол.
    correction_angle = -angle_before

    processed_sample = Rotate(correction_angle)(sample)

    # Площадь лучше считать после поворота:
    # интерполяция и дискретизация могут немного её изменить.
    area_after_rotation = int(
        torch.count_nonzero(processed_sample["mask"]).item()
    )

    if area_after_rotation == 0:
        raise ValueError(
            "После поворота маска стала пустой"
        )

    # Scale изменяет линейные размеры.
    # Площадь изменяется в scale ** 2 раз.
    scale = np.sqrt(etalon_area / area_after_rotation)

    processed_sample = Scale(scale)(processed_sample)

    angle_after = float(
        vertical_deviation(
            processed_sample["mask"],
            normalize=True,
            degrees=True,
        ).detach().cpu().item()
    )

    return processed_sample, angle_before, angle_after, float(scale)


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

    processed_samples: list[Sample] = []
    angles_before: list[float] = []
    angles_after: list[float] = []
    scales: list[float] = []

    for sample_id, sample in zip(sample_ids, samples):
        (
            processed_sample,
            angle_before,
            angle_after,
            scale,
        ) = sample_process(
            sample,
            etalon_area=3_000_000,
        )

        processed_samples.append(processed_sample)
        angles_before.append(angle_before)
        angles_after.append(angle_after)
        scales.append(scale)

        final_area = torch.count_nonzero(
            processed_sample["mask"]
        ).item()

        print(
            f"Sample {sample_id}: "
            f"angle_before={angle_before:.4f}°, "
            f"angle_after={angle_after:.4f}°, "
            f"scale={scale:.6f}, "
            f"final_area={final_area}"
        )

    processed_samples_numpy = [
        sample.to_numpy()
        for sample in processed_samples
    ]

    images = [
        to_image(sample)
        for sample in processed_samples_numpy
    ]

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(7 * len(images), 7),
        squeeze=False,
    )

    axes = axes[0]

    for (
        ax,
        image,
        sample_id,
        angle_before,
        angle_after,
        scale,
    ) in zip(
        axes,
        images,
        sample_ids,
        angles_before,
        angles_after,
        scales,
    ):
        h, w = image.shape[:2]

        ax.imshow(image)
        ax.set_aspect("equal")
        ax.set_title(
            f"Sample {sample_id}\n"
            f"angle: {angle_before:.2f}° → {angle_after:.2f}°\n"
            f"scale: {scale:.4f}"
        )
        ax.axis("off")

        # После исправления линия должна быть почти вертикальной.
        draw_deviation_axis(
            ax,
            angle_after,
            w,
            h,
        )

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