from pathlib import Path

import cv2
import numpy as np
from .shape_load import shape_load

def mask_load(
    path: Path,
    shape: tuple[int, ...],
) -> np.ndarray:
    lines, shape_type = shape_load(path)

    if shape_type not in {
        "POLYGON",
        "POLYGONZ",
        "POLYGONM",
    }:
        raise ValueError(
            f"Ожидался POLYGON, получен {shape_type}"
        )
    mask = mask_prepare(lines, shape)
    return mask


def mask_prepare(
    lines: list[np.ndarray],
    shape: tuple[int, ...]
) -> np.ndarray:
    height, width = shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        lines,
        255,
    )

    return mask


def label_load(
    path: Path,
    shape: tuple[int, ...],
    thickness: int = 1,
) -> np.ndarray:
    lines, shape_type = shape_load(path)

    if shape_type not in {
        "POLYLINE",
        "POLYLINEZ",
        "POLYLINEM",
    }:
        raise ValueError(
            f"Ожидался POLYLINE, получен {shape_type}"
        )

    label = label_prepare(
        lines=lines,
        shape=shape,
        thickness=thickness,
    )

    return label


def label_prepare(
    lines: list[np.ndarray],
    shape: tuple[int, ...],
    thickness: int = 1,
) -> np.ndarray:
    if thickness <= 0:
        raise ValueError(
            f"Толщина линии должна быть положительной: {thickness}"
        )

    height, width = shape[:2]

    label = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.polylines(
        label,
        lines,
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )

    return label