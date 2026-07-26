import cv2
from pathlib import Path
from utils import Dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch import nn
from trainer import Trainer
from adapter import NumpyAdapter
from cropper import Cropper

def train():

    device = "cuda"
    path_dataset = Path(r"D:\Data\dataset.lst")

    dataset = Dataset(
        path_dataset
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    state_dict = torch.load("saved_models/encoder_resnet34_channels3_depth5.pth", map_location=device)
    model = model.to(device)

    model.encoder.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    batch_size = 1
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    trainer = Trainer(model, dataloader, optimizer, device)


    test_image_folder = Path(r".\test_images\outcrop")
    train_test_folder = Path(r".\train_test")

    epochs = 100

    # начальное состояние модели
    save_test_images(model, test_image_folder, train_test_folder, 0)
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        trainer.train()
        save_test_images(model, test_image_folder, train_test_folder, epoch)
        torch.save(
            model.state_dict(),
            f"train_models_test/model_epoch_{epoch:03d}.pth"
        )


def save_test_images(model, test_image_folder: Path, train_test_folder: Path, epoch: int = 0):
    for _path in test_image_folder.iterdir():
        path_image = _path / (_path.name + ".png")
        print(path_image)
        epoch_folder = train_test_folder / Path(r"epoch_" + str(epoch))
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