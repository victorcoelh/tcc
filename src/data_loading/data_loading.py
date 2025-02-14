from pathlib import Path

import pandas as pd


def from_csv(csv_path: str, image_dir: str) -> tuple[list[str], list[list[str]]]:
    read_csv = pd.read_csv(csv_path)
    captions = read_csv["captions"]\
        .to_list()
        
    captions = list(map(fix_captions, captions))
    file_paths = read_csv["filepath"]\
        .map(lambda path: image_dir + path)\
        .to_list()
    
    return file_paths, captions


def from_cub(dataset_dir: Path, ids_path: Path,
             training: bool) -> tuple[list[str], list[list[str]]]:  # noqa: FBT001
    metadata = dataset_dir / "images.txt"
    image_folder = dataset_dir / "images/"
    descriptions_folder = dataset_dir / "descriptions/"

    paths: list[Path] = []
    descriptions: list[list[str]] = []

    for line in Path.open(metadata):
        _, image_path = line.strip("\n ").split(" ")
        paths.append(image_folder / image_path)
        caption_path = descriptions_folder / image_path

        caption = read_captions(caption_path.with_suffix(".txt"))
        descriptions.append(caption)
        
    with Path.open(ids_path) as f:
        should_keep = [should_get_sample(x[-2]=="1", training)
                       for x in f.readlines()]
        
    final_paths = [str(path) for path, keep in zip(paths, should_keep) if keep]
    descriptions = [captions for captions, keep in zip(descriptions, should_keep) if keep]
    
    return final_paths, descriptions


def should_get_sample(is_training_sample: bool, is_training_dataset: bool) -> bool:  # noqa: FBT001
    if is_training_dataset:
        return is_training_sample
    return not is_training_sample

        
def read_captions(file: Path) -> list[str]:
    with Path.open(file) as f:
        return f.readlines()


def fix_captions(captions: str) -> list[str]:
    fixed_captions = captions\
        .replace("\n", "")\
        .strip("[]")\
        .split("' '")
        
    return [caption.strip("'\"") for caption in fixed_captions]
