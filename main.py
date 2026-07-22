import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from rvm import HumanMatter


def main():
    image_path = r"D:\Data\humanspine\dataset\5\5_3.jpeg"
    checkpoint_path = r"./rvm/model_checkpoint/rvm_resnet50.pth"
    run_rvm(image_path, checkpoint_path)


def run_rvm(image_path: str, checkpoint_path: str, backbone: str = "resnet50"):
    # Загружаем изображение
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    matter = HumanMatter(checkpoint_path, "cuda")

    scale = 0.25

    image_resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    mask = matter(image_resized)

    # заполнение дыр внутри контура
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        mask,
        contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    mask = cv2.resize(
        mask,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_NEAREST
    )

    image[mask==0] = (0,0,0)

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