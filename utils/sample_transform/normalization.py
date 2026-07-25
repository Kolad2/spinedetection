from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from utils.tensor_inertia import vertical_deviation
from utils.sample import Sample


def calculate_normalization_parameters(
    sample: Sample,
    target_area: int = 3_000_000,
) -> tuple[float, float]:
    """
    Вычисляет параметры геометрической нормализации сэмпла.

    Parameters
    ----------
    sample:
        Сэмпл, содержащий бинарную маску в поле ``sample["mask"]``.

    target_area:
        Целевая площадь маски в пикселях после масштабирования.

    Returns
    -------
    scale:
        Линейный коэффициент масштабирования.

    rotation_angle:
        Угол, на который нужно повернуть изображение,
        чтобы главная ось маски стала вертикальной.

        Положительный угол — поворот против часовой стрелки.
        Отрицательный угол — поворот по часовой стрелке.
    """

    mask = sample["mask"]
    deviation_angle = vertical_deviation(
        mask,
        normalize=True,
        degrees=True,
    )
    current_area = torch.count_nonzero(mask).item()
    if current_area == 0:
        raise ValueError("Маска сэмпла пустая")
    scale = (target_area / current_area) ** 0.5
    rotation_angle = -float(deviation_angle.detach().cpu().item())
    return float(scale), rotation_angle


def calculate_transform_limits(
    sample: Sample,
    target_area: int = 3_000_000,
    scale_deviation: tuple[float, float] = (0.9, 1.1),
    angle_deviation: tuple[float, float] = (-5.0, 5.0),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Вычисляет диапазоны масштабирования и поворота исходного сэмпла,
    при которых результат будет находиться в заданных пределах
    относительно эталона.

    Parameters
    ----------
    sample:
        Исходный сэмпл с бинарной маской ``sample["mask"]``.

    target_area:
        Эталонная площадь маски в пикселях.

    scale_deviation:
        Допустимый итоговый линейный масштаб относительно эталона.

        Например:
            (0.9, 1.1)

        означает, что итоговый объект может быть от 90% до 110%
        эталонного линейного размера.

        Значение 1.0 соответствует точному эталонному размеру.

    angle_deviation:
        Допустимое итоговое отклонение от вертикали в градусах.

        Например:
            (-5.0, 5.0)

        означает допустимый наклон от -5° до +5° относительно
        вертикального эталона.

    Returns
    -------
    scale_range:
        Диапазон коэффициентов масштаба, которые нужно применять
        непосредственно к исходному изображению:

            (scale_min, scale_max)

    angle_range:
        Диапазон углов поворота исходного изображения:

            (angle_min, angle_max)

        Положительный угол — против часовой стрелки.
        Отрицательный угол — по часовой стрелке.
    """

    relative_scale_min, relative_scale_max = scale_deviation
    residual_angle_min, residual_angle_max = angle_deviation

    if relative_scale_min <= 0 or relative_scale_max <= 0:
        raise ValueError(
            "Границы scale_deviation должны быть больше нуля"
        )

    if relative_scale_min > relative_scale_max:
        raise ValueError(
            "Минимальный масштаб не может быть больше максимального"
        )

    if residual_angle_min > residual_angle_max:
        raise ValueError(
            "Минимальный угол не может быть больше максимального"
        )

    normalization_scale, normalization_angle = (
        calculate_normalization_parameters(
            sample,
            target_area=target_area,
        )
    )

    # normalization_scale приводит исходный объект точно к эталону.
    # Дополнительный множитель задаёт допустимое отклонение.
    scale_range = (
        normalization_scale * relative_scale_min,
        normalization_scale * relative_scale_max,
    )

    # normalization_angle приводит ось точно к вертикали.
    # К нему прибавляется допустимое остаточное отклонение.
    angle_range = (
        normalization_angle + residual_angle_min,
        normalization_angle + residual_angle_max,
    )

    return scale_range, angle_range