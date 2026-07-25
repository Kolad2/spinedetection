import math
from collections.abc import Iterator

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from utils.sample import Sample


class Tiler:
    def __init__(
        self,
        size: tuple[int, int],
        stride: tuple[int, int],
    ) -> None:
        """
        Parameters
        ----------
        size:
            Размер тайла в формате:

                (height, width)

        stride:
            Шаг тайлинга в формате:

                (stride_height, stride_width)
        """
        self.height, self.width = map(int, size)
        self.stride_height, self.stride_width = map(int, stride)

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"Размер тайла должен быть положительным: {size}"
            )

        if self.stride_height <= 0 or self.stride_width <= 0:
            raise ValueError(
                f"Шаг должен быть положительным: {stride}"
            )

        # Чтобы между тайлами не было пропусков.
        if self.stride_height > self.height:
            raise ValueError(
                "stride_height не должен быть больше высоты тайла"
            )

        if self.stride_width > self.width:
            raise ValueError(
                "stride_width не должен быть больше ширины тайла"
            )

    @staticmethod
    def _axis_layout(
        length: int,
        tile_size: int,
        stride: int,
    ) -> tuple[int, int, int, int]:
        """
        Вычисляет padding и количество тайлов вдоль одной оси.

        Returns
        -------
        padding_before:
            Padding слева или сверху.

        padding_after:
            Padding справа или снизу.

        padded_length:
            Итоговый размер оси после padding.

        tile_count:
            Количество тайлов вдоль оси.
        """
        if length <= tile_size:
            tile_count = 1
        else:
            tile_count = (
                math.ceil(
                    (length - tile_size) / stride
                )
                + 1
            )

        padded_length = (
            tile_size
            + (tile_count - 1) * stride
        )

        total_padding = padded_length - length

        padding_before = total_padding // 2
        padding_after = total_padding - padding_before

        return (
            padding_before,
            padding_after,
            padded_length,
            tile_count,
        )

    def get_padding(
        self,
        sample: Sample,
    ) -> tuple[int, int, int, int]:
        """
        Возвращает padding в формате:

            (left, top, right, bottom)
        """
        image_height, image_width = (
            sample["image"].shape[-2:]
        )

        top, bottom, _, _ = self._axis_layout(
            image_height,
            self.height,
            self.stride_height,
        )

        left, right, _, _ = self._axis_layout(
            image_width,
            self.width,
            self.stride_width,
        )

        return left, top, right, bottom

    def get_grid_size(
        self,
        sample: Sample,
    ) -> tuple[int, int]:
        """
        Возвращает количество тайлов:

            (rows, columns)
        """
        image_height, image_width = (
            sample["image"].shape[-2:]
        )

        _, _, _, rows = self._axis_layout(
            image_height,
            self.height,
            self.stride_height,
        )

        _, _, _, columns = self._axis_layout(
            image_width,
            self.width,
            self.stride_width,
        )

        return rows, columns

    def pad(
        self,
        sample: Sample,
    ) -> Sample:
        """
        Дополняет все поля Sample одинаковым нулевым padding.

        Исходный Sample не изменяется.
        """
        image_size = sample["image"].shape[-2:]
        left, top, right, bottom = self.get_padding(sample)

        padding = [
            left,
            top,
            right,
            bottom,
        ]

        padded: dict[str, torch.Tensor] = {}

        for name, tensor in sample.items():
            if tensor.shape[-2:] != image_size:
                raise ValueError(
                    f"Размер поля {name!r} не совпадает "
                    f"с изображением: "
                    f"{tuple(tensor.shape[-2:])} != "
                    f"{tuple(image_size)}"
                )

            result = F.pad(
                tensor,
                padding=padding,
                fill=0,
                padding_mode="constant",
            )

            # На случай, если конкретная версия torchvision
            # вернула обычный Tensor вместо TVTensor.
            if (
                isinstance(tensor, tv_tensors.TVTensor)
                and not isinstance(result, type(tensor))
            ):
                result = tv_tensors.wrap(
                    result,
                    like=tensor,
                )

            padded[name] = result

        return Sample(padded)

    @staticmethod
    def crop(
        sample: Sample,
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> Sample:
        """
        Вырезает одинаковую область из всех полей Sample.

        Интерполяция не выполняется.
        """
        cropped: dict[str, torch.Tensor] = {}

        for name, tensor in sample.items():
            result = tensor[
                ...,
                top:top + height,
                left:left + width,
            ]

            if (
                isinstance(tensor, tv_tensors.TVTensor)
                and not isinstance(result, type(tensor))
            ):
                result = tv_tensors.wrap(
                    result,
                    like=tensor,
                )

            cropped[name] = result

        return Sample(cropped)

    def iter_with_coordinates(
        self,
        sample: Sample,
    ) -> Iterator[tuple[Sample, int, int]]:
        """
        Выдаёт тайлы вместе с координатами на padded-холсте.

        Returns
        -------
        tile:
            Тайловый Sample.

        top:
            Верхняя координата тайла.

        left:
            Левая координата тайла.
        """
        padded_sample = self.pad(sample)

        padded_height, padded_width = (
            padded_sample["image"].shape[-2:]
        )

        for top in range(
            0,
            padded_height - self.height + 1,
            self.stride_height,
        ):
            for left in range(
                0,
                padded_width - self.width + 1,
                self.stride_width,
            ):
                tile = self.crop(
                    padded_sample,
                    top=top,
                    left=left,
                    height=self.height,
                    width=self.width,
                )

                yield tile, top, left

    def __call__(
        self,
        sample: Sample,
    ) -> Iterator[Sample]:
        """
        Дополняет Sample и последовательно выдаёт его тайлы.
        """
        for tile, _, _ in self.iter_with_coordinates(sample):
            yield tile