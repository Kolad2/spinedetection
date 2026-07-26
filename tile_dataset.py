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
    folder_rawdataset = Path(r"D:\Data\humanspine\dataset")
    folder_train_dataset = Path(r"./test")
    path_lst = folder_train_dataset / "train.lst"

    tiler = Tiler(
        size=(1024, 1024),
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
    suffix = f"original"
    tile_sample(sample, tiler, save_folder, suffix, lst)

    scale_range, angle_range = calculate_transform_limits(
        sample,
        scale_deviation=(0.7, 1.3),
        angle_deviation=(-10.0, 10.0),
    )
    for i in range(6):
        transformed_sample, scale, angle = get_random_affine_sample(
            sample,
            scale_range=scale_range,
            angle_range=angle_range,
            expand=True,
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