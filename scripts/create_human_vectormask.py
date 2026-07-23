import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from rvm import HumanMatter
from utils.manager_shapefile import mask_save


def main() -> None:
    folder_images = Path(
        r"D:\Data\humanspine\dataset"
    )
    checkpoint_path = Path(
        r"../rvm/model_checkpoint/rvm_resnet50.pth"
    )
    replace = True
    tolerance = 4
    scale = 0.25

    matter = HumanMatter(
        str(checkpoint_path),
        "cuda",
        threshold=0.5,
    )

    for person_folder in sorted(folder_images.iterdir()):
        if not person_folder.is_dir():
            continue

        name = person_folder.name

        image_path = person_folder / f"{name}_3.jpeg"
        vector_mask_path = (
            person_folder / f"{name}_3_vectormask"
        )

        if not image_path.is_file():
            print(f"Файл не найден: {image_path}")
            continue

        print(f"Обработка: {image_path}")

        try:
            run_rvm_masksaver(
                image_path=image_path,
                vector_mask_path=vector_mask_path,
                matter=matter,
                scale=scale,
                tolerance=tolerance,
                replace=replace
            )
        except Exception as error:
            print(
                f"Ошибка при обработке "
                f"{image_path}: {error}"
            )


def run_rvm_masksaver(
    image_path: Path,
    vector_mask_path: Path,
    matter: HumanMatter,
    tolerance: float = 1.0,
    scale=0.25,
    replace: bool = False,
) -> None:
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(
            f"Не удалось загрузить изображение: "
            f"{image_path}"
        )

    image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = image.shape[:2]

    image_resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    mask = matter(image_resized)

    mask, bbox = mask_correction(
        mask,
        copy=False,
    )

    print(f"BBox на уменьшенной маске: {bbox}")

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Сохранение:
    # имя/имя_3_vectormask/имя_3_vectormask.shp
    mask_save(
        vector_mask_path,
        mask,
        replace=replace,
        epsilon = tolerance
    )

    print(
        f"Shapefile сохранён: "
        f"{vector_mask_path}"
    )

    # testmask(image, mask)


def mask_correction(
    mask: np.ndarray,
    copy: bool = False,
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int] | None,
]:
    """
    Оставляет только самый большой связный объект
    и заполняет его значением 255.

    Если copy=False, исходный массив изменяется на месте.
    Если copy=True, возвращается обработанная копия.
    """
    if mask.dtype != np.uint8:
        raise TypeError(
            f"Ожидался uint8, получен {mask.dtype}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Ожидалась одноканальная маска HxW, "
            f"получена форма {mask.shape}"
        )

    mask = mask.copy() if copy else mask

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        mask.fill(0)
        return mask, None

    mask.fill(0)

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    bbox = cv2.boundingRect(
        largest_contour
    )

    return mask, bbox


def testmask(
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    image = image.copy()
    image[mask == 0] = (0, 0, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Alpha matte")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()