from collections.abc import Iterator

import cv2
import numpy as np

from src.utils.type_hints import Image, ImageBatch


def resize_images(images: Iterator[Image], new_size: int) -> ImageBatch:
    resized_images = [cv2.resize(img, (new_size, new_size)) for img in images]
    return np.stack(resized_images, dtype=np.uint8)
