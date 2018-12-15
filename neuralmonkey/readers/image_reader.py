from typing import Callable, Iterable, List
import os

import numpy as np
from typeguard import check_argument_types
from PIL import Image, ImageFile

from neuralmonkey.logging import warn


ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_reader(pad_w: int,
                 pad_h: int,
                 channels: int = 3,
                 prefix: str = "",
                 rescale_w: bool = False,
                 rescale_h: bool = False,
                 keep_aspect_ratio: bool = False,
                 mode: str = "RGB") -> Callable:
    """Get a reader of images loading them from a list of pahts.

    Args:
        pad_w: Width to which the images will be padded/cropped/resized.
        pad_h: Height to with the images will be padded/cropped/resized.
        channels: Number of channels in each image (default 3 for RGB)
        prefix: Prefix of the paths that are listed in a image files.
        rescale_w: If true, image is rescaled to have given width. It is
            cropped/padded otherwise.
        rescale_h: If true, image is rescaled to have given height. It is
            cropped/padded otherwise.
        keep_aspect_ratio: Flag whether the aspect ration should be kept during
            rescaling. Can only be used if both width and height are rescaled.
        mode: Scipy image loading mode, see scipy documentation for more
            details.

    Returns:
        The reader function that takes a list of image paths (relative to
        provided prefix) and returns a list of images as numpy arrays of shape
        pad_h x pad_w x number of channels.
    """
    check_argument_types()
    if not rescale_w and not rescale_h and keep_aspect_ratio:
        raise ValueError(
            "It does not make sense to keep the aspect ratio while not "
            "rescaling the image.")
    if rescale_w != rescale_h and not keep_aspect_ratio:
        raise ValueError(
            "While rescaling only one side, aspect ratio must be kept, "
            "was set to false.")

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
                        warn("Skipping image from file '{}' no. '{}'.".format(
                            path, i + 1))
                        image = Image.new(mode, (pad_w, pad_h))

                    image = _rescale_or_crop(image, pad_w, pad_h,
                                             rescale_w, rescale_h,
                                             keep_aspect_ratio)
                    image_np = np.array(image)

                    if len(image_np.shape) == 2:
                        img_channels = 1
                        image_np = np.expand_dims(image_np, 2)
                    elif len(image_np.shape) == 3:
                        img_channels = image_np.shape[2]
                    else:
                        raise ValueError(
                            ("Image should have either 2 (black and white) "
                             "or three dimensions (color channels), has {} "
                             "dimension.").format(len(image_np.shape)))

                    if channels != img_channels:
                        raise ValueError(
                            "Image does not have the pre-declared number of "
                            "channels {}, but {}.".format(
                                channels, img_channels))

                    yield _pad(image_np, pad_w, pad_h, channels)

    return load


# Mean pixel values from preprocessing of the VGG network
VGG_RGB_MEANS = [[[123.68, 116.779, 103.939]]]


def imagenet_reader(prefix: str,
                    target_width: int = 227,
                    target_height: int = 227,
                    vgg_normalization: bool = False,
                    zero_one_normalization: bool = False) -> Callable:
    """Load and prepare image the same way as Caffe scripts.

    The image preprocessing first rescales the image such that smaller edge has
    the target length. Then the middle rectangle is cropped from the resized
    image, such that the cropped image has the target size.

    Args:
        prefix: Prefix of the paths that are listed in a image files.
        target_width: Width of the image fed into an ImageNet network.
        target_height: Height of the image fed into an ImageNet network.
        vgg_normalization: If true, a mean pixel value will subtracted
            from all pixels. This is used for VGG nets.
        zero_one_normalization: If true, all pixel values are divided by 255
            such that they are in [0, 1] range. This is used for ResNet.

    Yield:
        An numpy array with the resized and cropped image for every image file
        in the list.
    """
    check_argument_types()

    def load(list_files: List[str]) -> Iterable[np.ndarray]:
        for list_file in list_files:
            with open(list_file) as f_list:
                for i, image_file in enumerate(f_list):
                    path = os.path.join(prefix, image_file.rstrip())

                    if not os.path.exists(path):
                        raise Exception(
                            "Image file '{}' no. {} does not exist."
                            .format(path, i + 1))

                    res = single_image_for_imagenet(
                        path, target_height, target_width,
                        vgg_normalization, zero_one_normalization)

                    yield res
    return load


def single_image_for_imagenet(
        path: str, target_height: int, target_width: int,
        vgg_normalization: bool, zero_one_normalization: bool) -> np.ndarray:
    image = Image.open(path).convert("RGB")

    width, height = image.size
    if width == height:
        _rescale_or_crop(image, target_width, target_height,
                         True, True, False)
    elif height < width:
        _rescale_or_crop(
            image,
            int(width * float(target_height) / height),
            target_height, True, True, False)
    else:
        _rescale_or_crop(
            image, target_width,
            int(height * float(target_width) / width),
            True, True, False)
    cropped_image = _crop(image, target_width, target_height)

    res = _pad(np.array(cropped_image),
               target_width, target_height, 3)
    assert res.shape == (target_width, target_height, 3)

    if vgg_normalization:
        res -= VGG_RGB_MEANS
    if zero_one_normalization:
        res /= 255.

    return res


def _rescale_or_crop(image: Image.Image, pad_w: int, pad_h: int,
                     rescale_w: bool, rescale_h: bool,
                     keep_aspect_ratio: bool) -> Image.Image:
    """Rescale and/or crop the image based on the rescale configuration."""
    orig_w, orig_h = image.size
    if orig_w == pad_w and orig_h == pad_h:
        return image

    if rescale_w and rescale_h and not keep_aspect_ratio:
        image = image.resize((pad_w, pad_h), Image.BILINEAR)
    elif rescale_w and rescale_h and keep_aspect_ratio:
        ratio = min(pad_h / orig_h, pad_w / orig_w)
        image = image.resize((int(orig_w * ratio), int(orig_h * ratio)))
    elif rescale_w and not rescale_h:
        orig_w, orig_h = image.size
        if orig_w != pad_w:
            ratio = pad_w / orig_w
            image = image.resize((pad_w, int(orig_h * ratio)))
    elif rescale_h and not rescale_w:
        orig_w, orig_h = image.size
        if orig_h != pad_h:
            ratio = pad_h / orig_h
            image = image.resize((int(orig_w * ratio), pad_h))
    return _crop(image, pad_w, pad_h)


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
