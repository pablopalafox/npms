import numpy as np
from skimage import io


def save_grayscale_image(filename, image_numpy):
    image_to_save = np.copy(image_numpy)
    image_to_save = (image_to_save * 255).astype(np.uint8)
    
    if len(image_to_save.shape) == 2:
        io.imsave(filename, image_to_save)
    elif len(image_to_save.shape) == 3:
        assert image_to_save.shape[0] == 1 or image_to_save.shape[-1] == 1
        io.imsave(filename, image_to_save)