from pathlib import Path

import torch
from torchvision.io import write_jpeg, write_png
from utils.sample import Sample

def save_tile(
    tile: Sample,
    output_folder: Path,
    tile_name: str,
) -> tuple[Path, Path]:
    """
    Сохраняет image, mask и label одного тайла.
    """
    image_folder = output_folder / "images"
    label_folder = output_folder / "labels"

    image_folder.mkdir(parents=False, exist_ok=True)
    label_folder.mkdir(parents=False, exist_ok=True)

    image_path = image_folder / f"{tile_name}.png"
    label_path = label_folder / f"{tile_name}.png"

    image = tile["image"].detach().cpu()
    label = tile["label"].detach().cpu()

    if image.dtype != torch.uint8:
        image = image.clamp(0, 255).to(torch.uint8)

    if label.dtype != torch.uint8:
        label = label.to(torch.uint8)

    if label.ndim == 2:
        label = label.unsqueeze(0)

    write_png(
        image,
        str(image_path),
    )

    write_png(
        label,
        str(label_path),
    )

    return image_path, label_path