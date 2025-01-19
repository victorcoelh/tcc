import math
from typing import TypeVar

import cv2
import numpy as np
import pandas as pd

from src.data_loading.preprocessing import resize_images
from src.utils.type_hints import Image, ImageBatch

TDataset = TypeVar("TDataset", bound="Dataset")


class Dataset:
    def __init__(
        self,
        image_paths: list[str],
        captions: list[list[str]],
        batch_size: int,
        seed: int | None = None) -> None:

        seed = np.random.default_rng().integers(0, 1000) if seed is None else seed
        np.random.default_rng(seed).shuffle(image_paths)
        np.random.default_rng(seed).shuffle(captions)
        
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.captions = captions
        
    def __getitem__(self, index: int) -> tuple[ImageBatch, list[list[str]]]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        
        image_paths = self.image_paths[start:end]
        captions = self.captions[start:end]
        
        images = map(load_image, image_paths)
        images = resize_images(images, 224)
        
        return images, captions
    
    def __len__(self) -> int:
        return math.ceil(len(self.image_paths) / self.batch_size)
    
    def __iter__(self) -> TDataset: # type: ignore
        self.__current = -1
        self.loaded_to_memory = 1000
        return self # type: ignore
    
    def __next__(self) -> tuple[ImageBatch, list[list[str]]]:
        self.__current += 1
        if self.__current < len(self):
            return self.__getitem__(self.__current)
        raise StopIteration
    

def get_path_and_captions(csv_path: str, image_dir: str) -> tuple[list[str], list[list[str]]]:
    read_csv = pd.read_csv(csv_path)
    
    captions = read_csv["captions"]\
        .to_list()

    captions = list(map(fix_captions, captions))

    file_paths = read_csv["filepath"]\
        .map(lambda path: image_dir + path)\
        .to_list()
    
    return file_paths, captions


def load_image(image_path: str) -> Image:
    image = cv2.imread(image_path)[:, :, ::-1]
    return image


def fix_captions(captions: str) -> list[str]:
    fixed_captions = captions\
        .replace("\n", "")\
        .strip("[]")\
        .split("' '")
        
    return [caption.strip("'\"") for caption in fixed_captions]
