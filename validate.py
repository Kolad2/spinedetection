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
from utils.manager_shapefile import mask_load


def main():
    with open(r"config.json", 'r', encoding='utf-8') as file:
        config = json.load(file)  # <- загрузка из файла
    folder_validation = Path(config["validation"])
    folder_save = Path(r".\train_validation")
    path_checkpoint = Path(r".\saved_models")
    device = "cuda"

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # можно None, т.к. веса будут перезаписаны
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(
        path_checkpoint,
        map_location=device,
        weights_only=True,  # если PyTorch >= 2.0
    )

    model.load_state_dict(checkpoint)
    model = model.to(device)

    validate(model, folder_validation, folder_save)



def validate(model, folder_validation: Path, folder_save: Path, epoch: int = 0):
    for _path in folder_validation.iterdir():
        path_image = _path / (_path.name + "_3.jpeg")
        path_mask = _path / (_path.name + "_3_vectormask")
        print(path_image)
        epoch_folder = folder_save / Path(r"epoch_" + str(epoch))
        epoch_folder.mkdir(parents=False, exist_ok=True)
        save_image_test(model, path_image, path_mask=path_mask, save_folder=epoch_folder)


def save_image_test(model, image_path: Path, path_mask=None, save_folder=None):
    save_folder = "train_test" if save_folder is None else str(save_folder)

    model.eval()

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    image = cv2.resize(image,None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if path_mask is not None:
        mask = mask_load(path_mask, (height, width))
        resized_height, resized_width = image.shape[:2]
        mask = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        image[mask==0] = 0

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