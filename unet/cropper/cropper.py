import numpy as np

from .pad_image import pad_image


class Cropper:
    def __init__(
        self,
        model,
        crop: int = 512,
        pad: int = 64,
        display: bool = False,
    ):
        self.crop = int(crop)
        self.pad = int(pad)
        self.display = display
        self.model = model

        self.image = None
        self.output = None
        self.progress = None
        self.sh = None

        if self.crop <= 0:
            raise ValueError("crop must be positive")

        if self.pad < 0:
            raise ValueError("pad must be non-negative")

        if self.crop <= 2 * self.pad:
            raise ValueError("crop must be greater than 2 * pad")

    def get_crop_edge(
        self,
        x: int,
        y: int,
    ) -> None:
        image_crop = self.image[
            y:y + self.crop,
            x:x + self.crop,
        ]

        output_crop = self.model(image_crop)

        if output_crop.shape[:2] != (self.crop, self.crop):
            raise ValueError(
                f"Model returned shape {output_crop.shape}, "
                f"expected spatial shape {(self.crop, self.crop)}"
            )

        output_y = y + self.pad
        output_x = x + self.pad

        valid_size = self.crop - 2 * self.pad

        self.output[
            output_y:output_y + valid_size,
            output_x:output_x + valid_size,
        ] = output_crop[
            self.pad:self.crop - self.pad,
            self.pad:self.crop - self.pad,
        ]

        if self.progress is not None:
            self.progress.update(1)

    def center_edges(self) -> None:
        step = self.crop - 2 * self.pad

        tiles_y = (self.sh[0] - 2 * self.pad) // step
        tiles_x = (self.sh[1] - 2 * self.pad) // step

        for i in range(tiles_y):
            y = i * step

            for j in range(tiles_x):
                x = j * step
                self.get_crop_edge(x, y)

    def _count_crops(self) -> int:
        step = self.crop - 2 * self.pad

        tiles_y = (self.sh[0] - 2 * self.pad) // step
        tiles_x = (self.sh[1] - 2 * self.pad) // step

        return tiles_y * tiles_x

    @staticmethod
    def _make_progress(
        total: int,
        enabled: bool,
        desc: str = "crop inference",
    ):
        if not enabled:
            return None

        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("tqdm is not installed; progress display disabled")
            return None

        return tqdm(
            total=total,
            desc=desc,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        original_height, original_width = image.shape[:2]

        padded_image, padding = pad_image(
            image,
            crop=self.crop,
            pad=self.pad,
        )

        top, bottom, left, right = padding

        self.image = padded_image
        self.sh = padded_image.shape

        self.output = np.zeros(
            padded_image.shape[:2],
            dtype=np.float32,
        )

        total = self._count_crops()

        self.progress = self._make_progress(
            total=total,
            enabled=self.display,
        )

        try:
            self.center_edges()
        finally:
            if self.progress is not None:
                self.progress.close()
                self.progress = None

        output = self.output[
            top:top + original_height,
            left:left + original_width,
        ]

        return output