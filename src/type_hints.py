import numpy as np
from jaxtyping import Float32

Image = Float32[np.ndarray, "channels height width"]
ImageBatch = Float32[Image, "batch"]
