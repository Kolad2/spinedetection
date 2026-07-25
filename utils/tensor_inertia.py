import torch

def vertical_deviation(
    mask: torch.Tensor,
    normalize: bool = True,
    degrees: bool = True,
) -> torch.Tensor:
    """
    Возвращает знаковое отклонение главной оси маски от вертикали.

    Система координат:
        dx > 0 — вниз;
        dy > 0 — вправо;
        dy < 0 — влево.

    Угол:
        0°   — вертикальная ось;
        > 0  — при движении вниз ось отклоняется вправо;
        < 0  — при движении вниз ось отклоняется влево.

    Диапазон:
        [-90°, 90°].
    """

    _, _, eigenvectors = ellipse(
        mask,
        normalize=normalize,
    )

    # Главная ось соответствует максимальному собственному значению.
    direction = eigenvectors[:, -1]

    # Собственный вектор может быть v или -v.
    # Ориентируем его всегда вниз: dx >= 0.
    if direction[0] < 0:
        direction = -direction

    dx = direction[0]  # вниз
    dy = direction[1]  # вправо

    angle = torch.atan2(dy, dx)

    if degrees:
        angle = torch.rad2deg(angle)

    return angle

def ellipse(
    mask: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Возвращает центр маски, собственные значения
    и направления главных осей.

    Returns
    -------
    centroid:
        Центр масс в порядке (x, y).

    eigenvalues:
        Собственные значения:
        [lambda_min, lambda_max].

    eigenvectors:
        Матрица формы (2, 2).

        eigenvectors[:, 0] — направление lambda_min;
        eigenvectors[:, 1] — направление lambda_max.

        Компоненты каждого направления заданы в порядке (dx, dy):
        dx — вниз по строкам;
        dy — вправо по столбцам.
    """

    inertia, centroid = tensor_inertia(
        mask,
        normalize=normalize,
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(inertia)

    return centroid, eigenvalues, eigenvectors

from typing import Literal

import torch


def tensor_inertia(
    mask: torch.Tensor,
    normalize: bool = True,
    center_method: Literal["mean", "bbox"] = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет матрицу центральных вторых моментов бинарной маски.

    Система координат:
        x — вниз по строкам матрицы;
        y — вправо по столбцам матрицы;
        начало координат — левый верхний угол.

    Parameters
    ----------
    mask:
        Тензор формы (H, W), (1, H, W) или (H, W, 1).
        Фон равен 0, объект — любому ненулевому значению.

    normalize:
        Если True, моменты делятся на площадь объекта.

    center_method:
        Способ вычисления центра:

        "mean":
            Центр масс — среднее координат всех пикселей объекта.

        "bbox":
            Центр ограничивающего прямоугольника:

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

    Returns
    -------
    inertia:
        Матрица формы (2, 2):

            [[J_x,  J_xy],
             [J_xy, J_y ]]

    centroid:
        Центр в порядке (x, y), где:
        x — строка;
        y — столбец.

        Форма: (2,).
    """

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(
                f"Ожидалась одноканальная маска, получена форма {mask.shape}"
            )

    if mask.ndim != 2:
        raise ValueError(
            f"Маска должна иметь форму (H, W), получена {mask.shape}"
        )

    x, y = torch.nonzero(mask, as_tuple=True)

    if x.numel() == 0:
        raise ValueError("Маска не содержит ненулевых пикселей")

    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)

    if center_method == "mean":
        x_center = x.mean()
        y_center = y.mean()

    elif center_method == "bbox":
        print(x.min(), x.max(), y.min(), y.max())
        x_center = (x.min() + x.max()) / 2
        y_center = (y.min() + y.max()) / 2

    else:
        raise ValueError(
            f"Неизвестный center_method={center_method!r}. "
            f"Допустимые значения: 'mean', 'bbox'"
        )

    dx = x - x_center
    dy = y - y_center

    j_x = torch.sum(dx * dx)
    j_y = torch.sum(dy * dy)
    j_xy = torch.sum(dx * dy)

    if normalize:
        area = x.numel()

        j_x = j_x / area
        j_y = j_y / area
        j_xy = j_xy / area

    inertia = torch.stack(
        (
            torch.stack((j_x, j_xy)),
            torch.stack((j_xy, j_y)),
        )
    )

    centroid = torch.stack((x_center, y_center))

    return inertia, centroid