import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from utils.tensor_inertia import vertical_deviation
from utils.sample import Sample
from utils.tailer import Tiler
from utils.tailer import save_tile


from utils.sample_transform import get_random_affine_sample, calculate_transform_limits
from utils.sample_transform import horizontal_flip

def main() -> None:
    path_config = Path("config.toml")
    with path_config.open("rb") as file:
        config = tomllib.load(file)
    folder_rawdataset = Path(config["rawdataset"])

    folder_train_dataset = Path(config["dataset"])
    path_lst = Path(config["dataset_lst"])

    tiler = Tiler(
        size=(512, 512),
        stride=(512, 512),
    )

    with path_lst.open("w", encoding="utf-8") as lst:
        for sample_path in tqdm(folder_rawdataset.iterdir()):
            sample_name = sample_path.name
            image_tile_folder = folder_train_dataset / sample_name
            image_tile_folder.mkdir(parents=False, exist_ok=True)

            sample_paths = sample_paths_from_folder(sample_path)
            sample = Sample.load_sample(sample_paths, thickness=5).crop_to_mask()
            sample_processing(sample, tiler, image_tile_folder,  lst)

def sample_processing(
    sample: Sample, tiler: Tiler, save_folder, lst
):
    scale_range, angle_range = calculate_transform_limits(
        sample,
        scale_deviation=(0.8, 1.1),
        angle_deviation=(-10.0, 10.0),
    )
    scale_range = (scale_range[0]*0.5, scale_range[1]*0.5)

    for i in range(6):
        transformed_sample, scale, angle = get_random_affine_sample(
            sample,
            scale_range=scale_range,
            angle_range=angle_range,
            expand=True,
        )
        mask = transformed_sample["mask"] != 0  # (H, W), bool
        image = transformed_sample["image"]  # (C, H, W)

        transformed_sample["image"] = image.masked_fill(
            ~mask.unsqueeze(0),
            0,
        )
        suffix = f"{scale:.2f}_{angle:.2f}"
        tile_sample(transformed_sample, tiler, save_folder, suffix, lst)


def tile_sample(sample: Sample, tiler, save_folder: Path, suffix, lst):
    save_folder_normal = save_folder / suffix
    save_folder_flipped = save_folder / (suffix + r"_flipped")
    if save_folder_normal.exists():
        return
    save_folder_normal.mkdir(parents=False, exist_ok=True)
    save_folder_flipped.mkdir(parents=False, exist_ok=True)

    for index, tile in enumerate(tiler(sample)):
        image_path, label_path = save_tile(tile, save_folder_normal, f"{index}")
        lst.write(
            f"{image_path.absolute().as_posix()} "
            f"{label_path.absolute().as_posix()}\n"
        )
        tile = horizontal_flip(tile)
        image_path, label_path = save_tile(tile, save_folder_flipped, f"{index}")
        lst.write(
            f"{image_path.absolute().as_posix()} "
            f"{label_path.absolute().as_posix()}\n"
        )

def sample_paths_from_folder(
    folder_sample: Path,
) -> dict[str, Path]:
    return {
        "image": folder_sample / f"{folder_sample.name}_3.jpeg",
        "mask": folder_sample / f"{folder_sample.name}_3_vectormask",
        "label": folder_sample / f"{folder_sample.name}_3_vector",
    }

if __name__ == "__main__":
    main()