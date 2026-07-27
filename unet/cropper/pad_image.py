import math
import numpy as np

def pad_image(
	image: np.ndarray,
	crop: int = 512,
	pad: int = 64,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
	height, width = image.shape[:2]
	step = crop - 2 * pad

	tiles_y = math.ceil(height / step)
	tiles_x = math.ceil(width / step)

	padded_height = tiles_y * step + 2 * pad
	padded_width = tiles_x * step + 2 * pad

	padding_y = padded_height - height
	padding_x = padded_width - width

	top = padding_y // 2
	bottom = padding_y - top

	left = padding_x // 2
	right = padding_x - left

	if image.ndim == 2:
		padding = (
			(top, bottom),
			(left, right),
		)
	else:
		padding = (
			(top, bottom),
			(left, right),
			(0, 0),
		)

	padded_image = np.pad(
		image,
		pad_width=padding,
		mode="constant",
		constant_values=0,
	)

	return padded_image, (top, bottom, left, right)