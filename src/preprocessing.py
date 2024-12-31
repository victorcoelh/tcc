from collections.abc import Iterator

import cv2
import numpy as np

from src.type_hints import ImageBatch


def resize_images(images: Iterator[np.ndarray], new_size: int) -> ImageBatch:
    resized_images = (cv2.resize(img, (new_size, new_size)) for img in images)
    return np.array(resized_images)
