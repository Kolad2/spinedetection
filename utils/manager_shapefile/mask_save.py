from pathlib import Path

import cv2
import numpy as np

from .shape_save import shape_polygon_save


def mask_save(
    path: Path,
    mask: np.ndarray,
) -> None:
    if mask.ndim != 2:
        raise ValueError(
            f"Ожидалась одноканальная маска, "
            f"получена форма {mask.shape}"
        )

    binary_mask = np.where(
        mask != 0,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    polygons = []

    for contour in contours:
        polygon = contour[:, 0, :]

        # Полигону нужны минимум три вершины.
        if len(polygon) < 3:
            continue

        polygons.append(polygon)

    shape_polygon_save(
        path,
        polygons,
    )