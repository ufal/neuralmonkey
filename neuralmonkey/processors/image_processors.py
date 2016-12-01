from typing import Callable, List, Optional
import os
from scipy import ndimage


def load_images(prefix="",
                pad_w: Optional[int]=None,
                pad_h: Optional[int]=None,
                mode: str='RGB') -> Callable:
    def load(file_names: List[List[str]]):
        images = []
        for i, image_file_list in enumerate(file_names):
            file_name = " ".join(image_file_list)
            path = os.path.join(prefix, file_name)

            if not os.path.exists(path):
                raise Exception(
                    ("Image file '{}' no."
                     "{}  does not exist.").format(path, i + 1))

        image_raw = ndimage.imread(path, mode=mode)

        return images
    return load
