"""Image cropping utilities for diffusion model preprocessing.

This module provides center cropping functionality that matches the
preprocessing used in ADM (Ablated Diffusion Model) and other diffusion
model codebases.

The center crop implementation uses a multi-scale approach to preserve
image quality while resizing to the target resolution.
"""

import numpy as np
from PIL import Image


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center crop an image to a square of the specified size.

    Implements a multi-scale resizing strategy from the ADM codebase:
    1. Repeatedly halve the image while it's >= 2x the target size (using BOX filter)
    2. Final resize to target size using BICUBIC interpolation
    3. Center crop to exact target dimensions

    This approach preserves more detail than direct resizing for large
    scale reductions, as each halving step properly averages pixels.

    Dimension Flow:
        Input:  (W_in, H_in) - original image dimensions
                     │
                     ▼
        While min(W, H) >= 2 * target:
            Halve both dimensions using BOX filter
                     │
                     ▼
        Scale to: (W_scaled, H_scaled) where min(W_scaled, H_scaled) = target
            Using BICUBIC interpolation
                     │
                     ▼
        Center crop: (target, target)

    Args:
        pil_image: Input PIL Image of any size and aspect ratio.
        image_size: Target size for the output square image.
            Both height and width will be exactly image_size pixels.

    Returns:
        Center-cropped PIL Image of size (image_size, image_size).

    Example:
        >>> img = Image.open("photo.jpg")  # e.g., 1920x1080
        >>> cropped = center_crop_arr(img, 256)  # -> 256x256
        >>> cropped.size
        (256, 256)

    References:
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # Multi-scale resize: reduce by half repeatedly while image is >= 2x target
    # Using BOX filter for accurate downsampling (averages pixels)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    # Scale so that the smaller dimension equals image_size
    # This preserves aspect ratio while ensuring we can center crop
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    # Convert to numpy array for center cropping
    # arr shape: (H, W, C) for color images, (H, W) for grayscale
    arr = np.array(pil_image)

    # Calculate crop offsets to center the crop
    crop_y = (arr.shape[0] - image_size) // 2  # Vertical offset
    crop_x = (arr.shape[1] - image_size) // 2  # Horizontal offset

    # Perform center crop and convert back to PIL Image
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])
