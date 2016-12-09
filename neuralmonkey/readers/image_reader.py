from typing import Callable, List, Optional
import os
import numpy as np
from scipy import ndimage


def image_reader(prefix="",
                 pad_w: Optional[int]=None,
                 pad_h: Optional[int]=None,
                 rescale: bool=False,
                 mode: str='RGB') -> Callable:

    def load(file_names: List[List[str]]):
        for i, image_file_list in enumerate(file_names):
            file_name = " ".join(image_file_list)
            path = os.path.join(prefix, file_name)

            if not os.path.exists(path):
                raise Exception(
                    ("Image file '{}' no."
                     "{}  does not exist.").format(path, i + 1))

            image = ndimage.imread(path, mode=mode)

            channels = image.shape[3] if len(image.shape) == 3 else 1
            if rescale:
                image = _rescale(image, pad_w, pad_h)
            else:
                image = _crop(image, pad_w, pad_h)

            yield _pad(image, pad_w, pad_h, channels)

    return load

def _rescale(image, pad_w, pad_h):
    orig_w, orig_h = image.shape[:2]
    width_ratio = pad_w / orig_w
    height_ratio = pad_h / orig_h
    zoom_ratio = min(1., width_ratio, height_ratio)

    if zoom_ratio < 1.:
        return ndimage.zoom(image, zoom_ratio)
    else:
        return image

def _crop(image, pad_w, pad_h):
    orig_w, orig_h = image.shape[:2]
    if orig_w > pad_w:
        diff = orig_w - pad_w
        half_diff = diff // 2
        image = image[:, half_diff + (diff % 2):-half_diff]
    if orig_h > pad_h:
        diff = orig_h - pad_h
        half_diff = diff // 2
        image = image[half_diff + (diff % 2):-half_diff, :]

    return image

def _pad(image, pad_w, pad_h, channels):
    img_h, img_w = image.shape[:2]

    image_padded = np.zeros((pad_h, pad_w, channels))
    image_padded[:img_h, :img_w, :] = image

    return image_padded
