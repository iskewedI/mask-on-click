from enum import Enum

class TABS(Enum):
    POINT = 1
    MASK = 2


# TYPES
class Setting(Enum):
    SaveOutput = "save_output"
    SaveParams = "save_params"
    UpscaleOutput = "upscale_output"
    Steps = "steps"
    InpaintingFill = "inpainting_fill"
    DenoisingStrength = "denoising_strength"
    ResizeMode = "resize_mode"
    InpaintFullRes = "inpaint_full_res"
    InpaintFullResPadding = "inpaint_full_res_padding"


INPAINTING_FILL_METHODS = ["fill", "original", "latent_noise", "latent_nothing"]
