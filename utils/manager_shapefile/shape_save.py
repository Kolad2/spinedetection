from pathlib import Path
import numpy as np
import shapefile


def shape_polyline_save(
		path: Path,
		polylines
) -> None:
	if path.is_dir():
		path.mkdir(parents=False, exist_ok=True)
		path = path / (path.stem + ".shp")
	else:
		path.parent.mkdir(parents=False, exist_ok=True)
	with shapefile.Writer(path, shapefile.POLYLINE) as shp:
		shp.field("NAME", "C")
		for polyline in polylines:
			shp.line([(polyline * [1, -1]).tolist()])
			shp.record("Polyline")


def shape_polygon_save(
	path: Path,
	polygons: list[np.ndarray],
) -> None:
	path = Path(path)

	if path.suffix.lower() == ".shp":
		if not path.parent.is_dir():
			raise FileNotFoundError(
				f"Родительский каталог не существует: "
				f"{path.parent}"
			)
	else:
		path.mkdir(
			parents=False,
			exist_ok=True,
		)
		path = path / f"{path.name}.shp"

	with shapefile.Writer(
			str(path),
			shapeType=shapefile.POLYGON,
	) as shp:
		shp.field("NAME", "C")

		for polygon in polygons:
			polygon = np.asarray(
				polygon,
				dtype=np.int32,
			)

			if len(polygon) < 3:
				continue

			polygon = polygon * [1, -1]

			if not np.array_equal(
					polygon[0],
					polygon[-1],
			):
				polygon = np.vstack(
					(polygon, polygon[0])
				)

			shp.poly([polygon.tolist()])
			shp.record("Polygon")