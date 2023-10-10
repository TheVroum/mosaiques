from PIL import Image
import numpy as np
from skimage import img_as_ubyte

def img_to_float(path: str) -> np.ndarray:
    image = Image.open(path)
    image_array = np.array(image)
    image_ubyte = img_as_ubyte(image_array)
    return image_ubyte.astype('float32')

__all__ = [
    "img_to_float",
]