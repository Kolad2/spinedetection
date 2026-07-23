from pathlib import Path

import cv2
import numpy as np

from .shape_save import shape_polygon_save


def mask_save(
    path: Path,
    mask: np.ndarray,
    replace: bool = False,
    epsilon: float = 0.5,
) -> None:
    if mask.ndim != 2:
        raise ValueError(
            f"Ожидалась одноканальная маска, "
            f"получена форма {mask.shape}"
        )

    if epsilon < 0:
        raise ValueError(
            f"epsilon не может быть отрицательным: {epsilon}"
        )

    binary_mask = np.where(
        mask != 0,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        if epsilon > 0:
            contour = cv2.approxPolyDP(
                contour,
                epsilon=epsilon,
                closed=True,
            )

        polygon = contour[:, 0, :]

        if len(polygon) < 3:
            continue

        polygons.append(polygon)

    shape_polygon_save(
        path,
        polygons,
        replace=replace,
    )