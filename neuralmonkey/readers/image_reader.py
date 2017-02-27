from typing import Callable, Iterable, List, Optional
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_reader(prefix="",
                 pad_w: Optional[int]=None,
                 pad_h: Optional[int]=None,
                 rescale: bool=False,
                 mode: str='RGB') -> Callable:
    """Get a reader of images loading them from a list of pahts.

    Args:
        prefix: Prefix of the paths that are listed in a image files.
        pad_w: Width to which the images will be padded/cropped/resized.
        pad_h: Height to with the images will be padded/corpped/resized.
        rescale: If true, bigger images will be rescaled to the pad_w x pad_h
            size. Otherwise, they will be cropped from the middle.
        mode: Scipy image loading mode, see scipy documentation for more
            details.

    Returns:
        The reader function that takes a list of image paths (relative to
        provided prefix) and returns a list of images as numpy arrays of shape
        pad_h x pad_w x number of channels.
    """

    def load(list_files: List[str]) -> Iterable[np.ndarray]:
        for list_file in list_files:
            with open(list_file) as f_list:
                for i, image_file in enumerate(f_list):
                    path = os.path.join(prefix, image_file.rstrip())

                    if not os.path.exists(path):
                        raise Exception(
                            ("Image file '{}' no."
                             "{}  does not exist.").format(path, i + 1))

                    try:
                        image = Image.open(path).convert(mode)
                    except IOError:
                        image = Image.new(mode, (pad_w, pad_h))

                    if rescale:
                        _rescale(image, pad_w, pad_h)
                    else:
                        image = _crop(image, pad_w, pad_h)
                    image_np = np.array(image)

                    if len(image_np.shape) == 2:
                        channels = 1
                        image_np = np.expand_dims(image_np, 2)
                    elif len(image_np.shape) == 3:
                        channels = image_np.shape[2]
                    else:
                        raise ValueError(
                            ("Image should have either 2 (black and white) "
                             "or three dimensions (color channels), has {} "
                             "dimension.").format(len(image_np.shape)))

                    yield _pad(image_np, pad_w, pad_h, channels)

    return load


def imagenet_reader(prefix: str,
                    target_width: int=227,
                    target_height: int=227) -> Callable:
    """Load and prepare image the same way as Caffe scripts."""
    def load(list_files: List[str]) -> Iterable[np.ndarray]:
        for list_file in list_files:
            with open(list_file) as f_list:
                for i, image_file in enumerate(f_list):
                    path = os.path.join(prefix, image_file.rstrip())

                    if not os.path.exists(path):
                        raise Exception(
                            "Image file '{}' no. {} does not exist."
                            .format(path, i + 1))

                    image = Image.open(path).convert('RGB')

                    width, height = image.size
                    if width == height:
                        _rescale(image, target_width, target_height)
                    elif height < width:
                        _rescale(image,
                                 int(width * float(target_height) / height),
                                 target_height)
                    else:
                        _rescale(image,
                                 target_width,
                                 int(height * float(target_width) / width))
                    cropped_image = _crop(image, target_width, target_height)

                    res = _pad(np.array(cropped_image),
                               target_width, target_height, 3)
                    assert res.shape == (target_width, target_height, 3)
                    yield res
    return load


def _rescale(image: Image.Image, pad_w: int, pad_h: int) -> None:
    orig_w, orig_h = image.size
    if orig_w != pad_w or orig_h != pad_h:
        image.thumbnail((pad_w, pad_h))


def _crop(image: Image.Image, pad_w: int, pad_h: int) -> Image.Image:
    orig_w, orig_h = image.size
    w_shift = max(orig_w - pad_w, 0) // 2
    h_shift = max(orig_h - pad_h, 0) // 2

    even_w = max(orig_w - pad_w, 0) % 2
    even_h = max(orig_h - pad_h, 0) % 2

    return image.crop(
        (w_shift, h_shift, orig_w - w_shift - even_w,
         orig_h - h_shift - even_h))


def _pad(image: np.ndarray, pad_w: int, pad_h: int,
         channels: int) -> np.ndarray:
    img_h, img_w = image.shape[:2]

    image_padded = np.zeros((pad_h, pad_w, channels))
    image_padded[:img_h, :img_w, :] = image

    return image_padded
