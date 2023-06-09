import logging
from typing import Dict
from src.utils.img_fn import b642MAT
from src.api import stable_diff
from cv2 import Mat

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "resize_mode": 0,
    "upscaling_resize": 1.5,
    "upscaler_1": "SwinIR_4x",
    # "upscaler_2": "Nearest",
    # "extras_upscaler_2_visibility": 0.25,
    "gfpgan_visibility": 0.9,
    "codeformer_visibility": 0.7,
    "codeformer_weight": 1,
}


def upscale_img(img, options: Dict = {}) -> Mat:
    logger.info("Upscaling image...")

    if not options:
        options = DEFAULT_OPTIONS.copy()
    else:
        options = {**DEFAULT_OPTIONS, **options}

    response = stable_diff.upscale(img, options)

    if "error" in response:
        logger.error("Error in the inpainting process")
        logger.error(response)
        return None

    logger.info("Image upscaled sucessfully")
    return b642MAT(response["image"])
