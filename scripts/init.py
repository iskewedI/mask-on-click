import modules.scripts as scripts
from src.web import create_ui, get_current_masks_img
from modules.processing import process_images, Processed
from modules.shared import opts
from PIL import Image
import numpy as np


class SegmentExtension(scripts.Script):
    def title(self):
        return "Segment on click"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        _, elements = create_ui(self, opts)

        return elements

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def run(self, p, angle, checkbox, *args, **kwargs):
        image_mask_np = get_current_masks_img()

        p.image_mask = Image.fromarray((image_mask_np * 255).astype(np.uint8))

        # TODO: get UI info through UI object angle, checkbox
        proc = process_images(p)
        # TODO: add image edit process via Processed object proc
        return proc


# def init_extension():
#     return [(ui_component, "Segment on click", "segment_on_click_tab")]


# script_callbacks.on_ui_tabs(init_extension)
