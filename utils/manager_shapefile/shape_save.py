from pathlib import Path
import shapefile


def shape_polyline_save(path: Path, polylines):
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