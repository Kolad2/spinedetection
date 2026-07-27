import cv2
import tomllib
from pathlib import Path
from unet.utils import Dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch import nn
from unet.trainer import Trainer
from unet.adapter import NumpyAdapter
from unet.cropper import Cropper
from validate import validate

def train():
    path_config = Path("config.toml")
    with path_config.open("rb") as file:
        config = tomllib.load(file)


    device = "cuda"
    path_dataset = Path(config["dataset_lst"])

    dataset = Dataset(
        path_dataset
    )

    batch_size = 1
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    state_dict = torch.load(r"saved_models/encoder_resnet34_channels3_depth5.pth", map_location=device)
    model = model.to(device)

    model.encoder.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    trainer = Trainer(model, dataloader, optimizer, device)


    folder_validation = Path(config["validation"])
    train_test_folder = Path(r".\train_validation")

    epochs = 100

    # начальное состояние модели
    validate(model, folder_validation, train_test_folder, 0)
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        trainer.train()
        validate(model, folder_validation, train_test_folder, epoch)
        torch.save(
            model.state_dict(),
            f"train_models/model_epoch_{epoch:03d}.pth"
        )


def load_resnet_imagenet_encoder():
    encoder = smp.encoders.get_encoder(
        "resnet34",
        in_channels=3,
        depth=5,
        weights="imagenet",  # скачает веса
    )
    torch.save(encoder.state_dict(), "saved_models/encoder_resnet34_channels3_depth5.pth")


if __name__ == "__main__":
    train()