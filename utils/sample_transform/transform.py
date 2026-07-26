import math

import torch

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as transforms

from utils.sample import Sample


def horizontal_flip(sample: Sample) -> Sample:
    """
    Отражает по горизонтали все поля Sample:
    image, mask и label.
    """
    return Sample({
        name: F.horizontal_flip(tensor)
        for name, tensor in sample.items()
    })

class Rotate:
    def __init__(self, angle: float) -> None:
        """
        angle:
            Угол поворота torchvision в градусах.

            Положительный — против часовой стрелки.
            Отрицательный — по часовой стрелке.
        """
        self.angle = float(angle)

    def __call__(self, sample: Sample) -> Sample:
        rotate = transforms.RandomRotation(
            degrees=(self.angle, self.angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=True,
            fill={
                tv_tensors.Image: 0,
                tv_tensors.Mask: 0,
            },
        )

        rotated = rotate(dict(sample))

        return Sample(rotated)


class Scale:
    def __init__(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError(
                f"scale должен быть больше 0, получено {scale}"
            )

        self.scale = float(scale)

    def __call__(self, sample: Sample) -> Sample:
        height, width = sample["image"].shape[-2:]

        new_size = (
            max(1, round(height * self.scale)),
            max(1, round(width * self.scale)),
        )

        resize = transforms.Resize(
            size=new_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        resized = resize(dict(sample))

        return Sample(resized)


class RotateScale:
    """
    Одновременно поворачивает и масштабирует сэмпл.

    Поворот и масштабирование выполняются одной аффинной
    операцией, поэтому для каждого поля происходит только
    одна интерполяция.
    """

    def __init__(
        self,
        angle: float,
        scale: float,
        expand: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        angle:
            Угол поворота в градусах.

            Положительный — против часовой стрелки.
            Отрицательный — по часовой стрелке.

        scale:
            Линейный коэффициент масштабирования.

        expand:
            Если False, размер холста не изменяется.

            Если True, размер холста подгоняется так,
            чтобы вместить повёрнутое и масштабированное
            исходное изображение.
        """
        if scale <= 0:
            raise ValueError(
                f"scale должен быть больше 0, получено {scale}"
            )

        self.angle = float(angle)
        self.scale = float(scale)
        self.expand = bool(expand)

    def _output_size(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        """
        Вычисляет размер холста после поворота и масштабирования.

        Returns
        -------
        output_height, output_width
        """
        theta = math.radians(self.angle)

        cos_theta = abs(math.cos(theta))
        sin_theta = abs(math.sin(theta))

        output_width = math.ceil(
            self.scale
            * (
                width * cos_theta
                + height * sin_theta
            )
        )

        output_height = math.ceil(
            self.scale
            * (
                height * cos_theta
                + width * sin_theta
            )
        )

        return (
            max(1, output_height),
            max(1, output_width),
        )

    @staticmethod
    def _interpolation(
        tensor: torch.Tensor,
    ) -> InterpolationMode:
        """
        Выбирает интерполяцию в зависимости от типа поля.
        """
        if isinstance(tensor, tv_tensors.Mask):
            return InterpolationMode.NEAREST

        return InterpolationMode.BILINEAR

    def __call__(
        self,
        sample: Sample,
    ) -> Sample:
        height, width = sample["image"].shape[-2:]

        if self.expand:
            output_height, output_width = self._output_size(
                height,
                width,
            )
        else:
            output_height = height
            output_width = width

        # Рабочий холст должен быть не меньше и исходного,
        # и требуемого выходного холста.
        #
        # Если scale < 1, сначала affine выполняется на исходном
        # холсте, а затем результат обрезается без интерполяции.
        work_height = max(height, output_height)
        work_width = max(width, output_width)

        padding_height = work_height - height
        padding_width = work_width - width

        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top

        # torchvision padding:
        # [left, top, right, bottom]
        padding = [
            padding_left,
            padding_top,
            padding_right,
            padding_bottom,
        ]

        transformed: dict[str, torch.Tensor] = {}

        for name, tensor in sample.items():
            if tensor.shape[-2:] != (height, width):
                raise ValueError(
                    f"Размер поля {name!r} не совпадает "
                    f"с размером изображения: "
                    f"{tuple(tensor.shape[-2:])} != "
                    f"{(height, width)}"
                )

            # Padding не интерполирует изображение.
            padded = F.pad(
                tensor,
                padding=padding,
                fill=0,
                padding_mode="constant",
            )

            # Поворот и масштаб выполняются совместно:
            # это единственная интерполяция.
            result = F.affine(
                padded,
                angle=self.angle,
                translate=[0, 0],
                scale=self.scale,
                shear=[0.0, 0.0],
                interpolation=self._interpolation(tensor),
                fill=0,
                center=None,
            )

            # Обрезка не интерполирует изображение.
            if (
                work_height != output_height
                or work_width != output_width
            ):
                top = (work_height - output_height) // 2
                left = (work_width - output_width) // 2

                result = F.crop(
                    result,
                    top=top,
                    left=left,
                    height=output_height,
                    width=output_width,
                )

            transformed[name] = result

        return Sample(transformed)

def get_random_affine_sample(
    sample: Sample,
    scale_range: tuple[float, float],
    angle_range: tuple[float, float],
    expand: bool = True,
) -> tuple[Sample, float, float]:
    """
    Случайно поворачивает и масштабирует сэмпл.

    Поворот и масштабирование выполняются одной аффинной
    операцией, поэтому происходит одна интерполяция.

    Parameters
    ----------
    sample:
        Исходный сэмпл.

    scale_range:
        Диапазон коэффициента масштаба:

            (scale_min, scale_max)

    angle_range:
        Диапазон угла поворота в градусах:

            (angle_min, angle_max)

        Положительный угол — против часовой стрелки.
        Отрицательный угол — по часовой стрелке.

    expand:
        Если True, холст изменяется, чтобы вместить результат.
        Если False, размер холста остаётся исходным.

    Returns
    -------
    transformed_sample:
        Преобразованный сэмпл.
    """
    scale_min, scale_max = map(float, scale_range)

    if scale_min <= 0 or scale_max <= 0:
        raise ValueError(
            "Обе границы scale_range должны быть больше нуля"
        )

    scale = random_uniform(scale_range)
    angle = random_uniform(angle_range)

    transformed_sample = RotateScale(
        angle=angle,
        scale=scale,
        expand=expand,
    )(sample)

    return transformed_sample, scale, angle


def random_uniform(
    value_range: tuple[float, float],
) -> float:
    """
    Возвращает равномерно распределённое случайное значение
    из диапазона [minimum, maximum].
    """
    minimum, maximum = map(float, value_range)

    if minimum > maximum:
        raise ValueError(
            f"Левая граница диапазона больше правой: {value_range}"
        )

    if minimum == maximum:
        return minimum

    random_value = torch.rand(()).item()

    return minimum + random_value * (maximum - minimum)