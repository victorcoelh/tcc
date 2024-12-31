import numpy as np
from jaxtyping import UInt8

Image = UInt8[np.ndarray, "channels height width"]
ImageBatch = UInt8[Image, "batch"]
