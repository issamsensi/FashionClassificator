from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


def _to_centered_28x28(gray_array: np.ndarray) -> np.ndarray:
    mask = gray_array > 20
    if np.any(mask):
        rows, cols = np.where(mask)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        cropped = gray_array[y_min : y_max + 1, x_min : x_max + 1]
    else:
        cropped = gray_array

    cropped_img = Image.fromarray(cropped, mode="L")
    target_size = 20
    scale = min(target_size / cropped_img.width, target_size / cropped_img.height)
    new_w = max(1, int(round(cropped_img.width * scale)))
    new_h = max(1, int(round(cropped_img.height * scale)))
    resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(resized, (left, top))
    return np.array(canvas, dtype=np.uint8)


def preprocess_pil_image(image: Image.Image, invert_mode: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Convert uploaded image to Fashion-MNIST input shape and normalized values.

    Args:
        image: PIL image uploaded by user.
        invert_mode: one of "auto", "always", "never".

    Returns:
        model_input: float32 array with shape (1, 28, 28, 1), values in [0, 1]
        processed_2d: uint8 array with shape (28, 28) for UI preview
    """
    gray_image = ImageOps.grayscale(image)
    gray_array = np.array(gray_image, dtype=np.uint8)

    if invert_mode not in {"auto", "always", "never"}:
        raise ValueError("invert_mode must be one of: auto, always, never")

    should_invert = invert_mode == "always" or (
        invert_mode == "auto" and float(gray_array.mean()) > 127.0
    )
    if should_invert:
        gray_array = 255 - gray_array

    processed_2d = _to_centered_28x28(gray_array)
    model_input = processed_2d.astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)

    return model_input, processed_2d


def preprocess_image_path(image_path: str, invert_mode: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        return preprocess_pil_image(image, invert_mode=invert_mode)
