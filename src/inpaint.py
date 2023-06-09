import logging
import json
from src.api.stable_diff import stable_diff
from src.utils.static import INPAINTING_FILL_METHODS

logger = logging.getLogger(__name__)


default_options = {
    "inpaint_full_res": True,
    "sampler_name": "DPM++ 2S a Karras",
    "cfg_scale": 7,
    "inpainting_fill": 1,
    "steps": 35,
    "resize_mode": 0,
    "mask_blur": 4,
    "denoising_strength": 0.7,
    "restore_faces": True,
}


def inpaint(
    prompt,
    negative_prompt,
    img_b64,
    mask_b64,
    config,
):
    prompt = {
        "text": f"masterpiece, realistic, HD, 4K (({prompt}))",
        "negative": f"easynegative, ng_deepnegative_v1_75t, (({negative_prompt})), bad quality, malformed, ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, unclear",
    }

    api_config = {}
    api_config.update(
        {
            "init_images": [img_b64],
            "mask": mask_b64,
            "prompt": prompt["text"],
            "negative_prompt": prompt["negative"],
        }
    )
    api_config.update(default_options)
    api_config.update(config)

    logger.info(
        f"""
            - Steps: {api_config.get("steps")}
            - Mask blur: {api_config.get("mask_blur")}
            - Denoising strenght: {api_config.get("denoising_strength")}
            - CFG Scale: {api_config.get("cfg_scale")}
            - Inpainting fill: {INPAINTING_FILL_METHODS[api_config.get("inpainting_fill")]}
            - Image shapes: Width: {api_config.get("width")} | Height: {api_config.get("height")}
            - Prompt: {prompt}
            - Negative prompt: {negative_prompt}
            """
    )

    logger.info("--- Making request to API")
    response = stable_diff.img2img(api_config)

    if "error" in response:
        logger.exception("Error in the inpainting process")
        logger.exception(response)
        return None

    info = response.get("info")
    if info is not None:
        deserealized = json.loads(info)

        return {"seed": deserealized.get("seed"), "imgb64": response["images"][0]}
