from pathlib import Path

import numpy as np
import shapefile


def shape_load(
    path: Path,
) -> tuple[list[np.ndarray], str]:
    path = Path(path)

    if path.is_dir():
        shp_files = list(path.glob("*.shp"))

        if not shp_files:
            raise FileNotFoundError(
                f"В директории нет .shp файлов: {path}"
            )

        path = shp_files[0]

    if not path.is_file():
        raise FileNotFoundError(
            f"Файл не найден: {path}"
        )

    lines: list[np.ndarray] = []

    with shapefile.Reader(str(path)) as shp:
        shape_type = shapefile.SHAPETYPE_LOOKUP[
            shp.shapeType
        ]

        for shape in shp.shapes():
            line = np.asarray(
                shape.points,
                dtype=np.float64,
            )

            if len(line) <= 1:
                continue

            line[:, 1] *= -1

            lines.append(
                line.astype(np.int32)
            )

    return lines, shape_type