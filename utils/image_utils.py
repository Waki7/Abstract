from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils.model_utils as model_utils


def interpolate(image: Union[torch.Tensor, np.ndarray], target_size: tuple, interpolation_method: str):
    if interpolation_method.startswith('TORCH'):
        if interpolation_method.endswith('MAX'):
            image = F.adaptive_max_pool2d(input=torch.tensor(image), output_size=target_size).numpy()
        else:
            image = F.adaptive_avg_pool2d(input=torch.tensor(image), output_size=target_size).numpy()
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = cv2.resize(image, dsize=target_size, interpolation=getattr(cv2, interpolation_method))
        image = np.transpose(image, axes=(2, 0, 1)) if len(image.shape) == 3 else np.expand_dims(image, axis=0)
    return image


# ---------------------------------------------------------------------------
# IMAGES
# ---------------------------------------------------------------------------

def convert_to_rgb_format(image_array: Union[np.ndarray, torch.Tensor],
                          target_resolution=None, interpolation=cv2.INTER_NEAREST):
    image_array = model_utils.scale_vector_to_range(image_array, 0, 255).astype(np.uint8)
    assert image_array.shape[0] == 1 or image_array.shape[0] == 3, 'include the channel dimension'
    if image_array.shape[0] == 1:
        image_array = np.repeat(image_array, repeats=3, axis=0)
    if target_resolution is not None:  # resize
        return interpolate(image_array, target_size=target_resolution, interpolation_method=interpolation)
    return image_array
    # return Image.fromarray(image_vector)
