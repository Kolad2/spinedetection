from pathlib import Path

import cv2
import numpy as np

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

    height, width = shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if lines:
        cv2.fillPoly(
            mask,
            lines,
            255,
        )

    return mask