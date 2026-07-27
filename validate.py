import cv2
from pathlib import Path
from unet.utils import Dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import json
from torch import nn
from unet.trainer import Trainer
from unet.adapter import NumpyAdapter
from unet.cropper import Cropper


def main():
    with open(r"config.json", 'r', encoding='utf-8') as file:
        config = json.load(file)  # <- загрузка из файла
    test_image_folder = Path(r".\test_images\outcrop")
    train_test_folder = Path(r".\train_test")



def validate(model, folder_validation: Path, folder_save: Path, epoch: int = 0):
    for _path in folder_validation.iterdir():
        path_image = _path / (_path.name + ".png")
        print(path_image)
        epoch_folder = folder_save / Path(r"epoch_" + str(epoch))
        epoch_folder.mkdir(parents=False, exist_ok=True)
        save_image_test(model, path_image, save_folder=epoch_folder)


def save_image_test(model, image_path: Path, save_folder=None):
    save_folder = "train_test" if save_folder is None else str(save_folder)

    model.eval()

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = NumpyAdapter(model)
    model = Cropper(model, 512, 64, display=True)
    result = model(image)

    fig = plt.figure(figsize=(10, 4))
    axs = [
        fig.add_subplot(1, 2, 1),
        fig.add_subplot(1, 2, 2)
    ]

    axs[0].imshow(image)
    axs[1].imshow(result, cmap="gray")
    # plt.show()
    fig.savefig(save_folder + "/" + image_path.name)
    plt.close(fig)


if __name__ == "__main__":
    main()