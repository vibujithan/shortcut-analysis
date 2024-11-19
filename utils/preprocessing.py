import os
from math import ceil
from pathlib import Path
from torchvision import transforms

import nibabel as nib
import numpy as np
import tifffile.tifffile as tiff
from scipy.ndimage import zoom
from tqdm.auto import tqdm





def minmax(image):
    image = image.astype('f8')
    maxv = np.amax(image)
    minv = np.amin(image)
    return ((image - minv) / maxv).astype('f4')


def center_crop(image, size):
    original_size = image.shape
    x_init = ceil((original_size[0] - size[0]) / 2)
    y_init = ceil((original_size[1] - size[1]) / 2)
    z_init = ceil((original_size[2] - size[2]) / 2)
    image = image[x_init:x_init + size[0], y_init:y_init + size[1], z_init:z_init + size[2]]
    return image


def resize(image, size):
    return zoom(image, size)


if __name__ == '__main__':
    main()
