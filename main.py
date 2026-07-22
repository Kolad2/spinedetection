import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from rvm import HumanMatter


def main():
    folder_images = Path(r"D:\Data\humanspine\dataset")
    checkpoint_path = r"./rvm/model_checkpoint/rvm_resnet50.pth"

    matter = HumanMatter(checkpoint_path, "cuda", threshold=0.75)

    for person_folder in sorted(folder_images.iterdir()):
        if not person_folder.is_dir():
            continue

        # Например: папка 5 -> файл 5_3.jpeg
        image_path = person_folder / f"{person_folder.name}_3.jpeg"

        if not image_path.exists():
            print(f"Файл не найден: {image_path}")
            continue

        print(f"Обработка: {image_path}")

        try:
            run_rvm(str(image_path), matter)
        except Exception as error:
            print(f"Ошибка при обработке {image_path}: {error}")


def run_rvm(image_path: str, matter: HumanMatter):
    # Загружаем изображение
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    scale = 0.25

    image_resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    mask = matter(image_resized)

    mask, bbox = mask_correction(mask, copy=False)

    print(bbox)

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    testmask(image, mask)

def mask_correction(
    mask: np.ndarray,
    copy: bool = False,
) ->  tuple[np.ndarray, tuple]:
    """
    Оставляет только самый большой связный объект и заполняет его значением 255.

    Если copy=False, исходный массив изменяется на месте.
    Если copy=True, возвращается обработанная копия.
    """
    if mask.dtype != np.uint8:
        raise TypeError(f"Ожидался uint8, получен {mask.dtype}")

    if mask.ndim != 2:
        raise ValueError(
            f"Ожидалась одноканальная маска HxW, получена форма {mask.shape}"
        )

    mask = mask.copy() if copy else mask

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask

    # Удаляем все объекты
    mask.fill(0)



    largest_contour = max(contours, key=cv2.contourArea)

    # Рисуем только самый большой объект без дыр
    cv2.drawContours(
        mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    bbox = cv2.boundingRect(largest_contour)

    return mask, bbox


def testmask(image, mask):
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

    plt.savefig(
        "result.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":
    main()