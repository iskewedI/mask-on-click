import logging
import time
import numpy as np
import cv2
import gradio as gr
from src.utils import gradio_fn, masks_fn, img_fn, thread_fn
from src.utils.static import TABS
from src.api import stable_diff
from src.inpaint import inpaint
from src.segmentator import (
    gen_image_embedding,
    segment_ONNX_point,
    segment_automatic,
    segment_boxes,
)
from src.upscale import upscale_img
import os
from src.UI import UIConfig
from src.utils.static import Setting
from src.utils.logging import config_logs
import modules.scripts as scripts
import PIL

config_logs()
logger = logging.getLogger(__name__)


# Configuration
styles_css = os.path.join(scripts.basedir(), "styles.css")

UIConfig.load_params()

# Global variables
# Tabs
current_tab = TABS.POINT

# Masks - Masks tab
segmented_masks: list = []
selected_mask_indices: list = []
multimask_output = True

# Masks - Points tab
drawn_masks_per_id: dict = {}  # dict<str, {"mask": [], "last_color": []}>

# Image copies in local
original_image = None
image_segmentated = None
img_embedding = None

out_path = os.path.join(scripts.basedir(), "out")
if not os.path.exists(out_path):
    os.mkdir(out_path)

# SD Inpaint input data
inpaint_input_data = {
    "prompt": UIConfig.get_param("prompt"),
    "negative_prompt": UIConfig.get_param("prompt_negative"),
}

tmp_img_path = os.path.join(scripts.basedir(), "temp.png")

def get_current_masks_img():
    masks = None

    if current_tab == TABS.POINT:
        masks = [m["mask"] for m in drawn_masks_per_id.values()]
    elif current_tab == TABS.MASK:
        masks = [
            value
            for index, value in enumerate(segmented_masks)
            if index in selected_mask_indices
        ]

    if len(masks) == 0:
        logger.warning("No masks selected. Please segment the image.")
        return

    logger.info("Joining masks...")
    joined_masks = masks_fn.join_masks_img(masks)

    return joined_masks
# Set globals to original values
def clean_globals():
    global segmented_masks
    global selected_mask_indices
    global drawn_masks_per_id
    global image_segmentated
    global img_embedding
    global original_image

    # Reset masks vars
    segmented_masks = []
    selected_mask_indices = []
    drawn_masks_per_id = {}

    # Reset image vars
    image_segmentated = None
    img_embedding = None
    original_image = None


# Changes the current tab
def set_current_tab(tab: TABS):
    global current_tab

    current_tab = tab


# Get current SD operation progress
def get_sd_progress():
    result = stable_diff.get_progress()

    if "error" in result:
        logger.exception("Failed to get current progress => ", result)
        return None

    if not result["current_image"]:
        return None

    current_img = img_fn.b642img(result["current_image"])

    return current_img


# Update the mask overlay on the image
def update_mask_overlay(img, color=[152, 251, 152], alpha=0.6):
    global selected_mask_indices
    global segmented_masks

    # Filter only selected masks
    selected_masks = [
        mask
        for index, mask in enumerate(segmented_masks)
        if index in selected_mask_indices
    ]

    if len(selected_masks) == 0:
        # No mask selected
        return img

    # Draw the selected mask indices on the overlay image
    overlay_image = masks_fn.get_masks_img(
        img, selected_masks, np.array(color) * alpha, 1
    )

    return overlay_image


# Handlers
# Handle the user's click on the image
def handle_mask_selection(evt: gr.SelectData):
    global selected_mask_indices
    global segmented_masks
    global image_segmentated

    selected_mask = masks_fn.get_nearest_mask_index(
        segmented_masks, evt.index[0], evt.index[1]
    )

    if selected_mask is not None:
        mask_index = selected_mask["index"]
        if mask_index in selected_mask_indices:
            selected_mask_indices.remove(mask_index)
        else:
            selected_mask_indices.append(mask_index)

    return update_mask_overlay(image_segmentated)


# Function executed on the image upload event
def handle_image_upload(img, convert_color=True):
    global original_image
    global img_embedding

    input_img = img.copy()

    if convert_color:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if original_image is None or not img_fn.is_similar(original_image, input_img):
        # New image uploaded
        # Clean globals
        clean_globals()

        # Save original image copy
        original_image = input_img

        img_embedding = gen_image_embedding(input_img)

    return input_img


# Handle the "send to input" button clicks
def handle_send_to_input(out_img):
    # We don't want to convert colors in this case as they're already in correct form
    handle_image_upload(out_img, False)

    return out_img, out_img


def create_ui(segment_script, opts):
    with gr.Blocks(css=styles_css) as demo:
        # with gr.Row():
        # get_img_from_inpaint = gr.Button("Get image from inpaint")

        ## Tabs
        with gr.Tab("Point segmentation") as points_tab:
            image_clicker = gr.Image(
                tool=None,
                elem_classes="gr-img",
                elem_id="gr-clicker-image",
                label="Select point",
            ).style(height=opts.img2img_editor_height)

        with gr.Tab("Mask segmentation") as masks_tab:
            with gr.Row():
                multimask_output_check = gr.Checkbox(
                    multimask_output, label="Multi mask output"
                )
            # Masking
            with gr.Row():
                with gr.Column():
                    image_masker = gr.ImageMask(
                        label="Mask image",
                        type="numpy",
                        brush_radius=10,
                        elem_classes="gr-img",
                    ).style(height=720)
                    segment_button = gr.Button("Segment")

                with gr.Column():
                    with gr.Row():
                        segmented_image = gr.Image(tool=None, label="Segmentations")
                    with gr.Row():
                        segmented_boxs_image = gr.Image(
                            label="Boxes generated",
                            elem_classes="gr-img-segm-result-boxes",
                        )

        ## Inpainting results
        # with gr.Column():
        # Internal settings
        # config_elements = UIConfig.get_ui()

        # Text settings
        # with gr.Row():
        #     prompt_text = gr.Textbox(
        #         inpaint_input_data["prompt"],
        #         label="Prompt",
        #     )
        #     prompt_negative = gr.Textbox(
        #         inpaint_input_data["negative_prompt"], label="Negative prompt"
        #     )

        # inpaint_button = gr.Button("Inpaint", variant="primary")

        # result_image = gr.Image(
        #     label="Output",
        #     elem_classes="gr-img",
        #     interactive=False,
        # )

        # send_out_to_in_button = gr.Button("Send to input")

        # Handles when the user clicks on the image on the click segmentation tab

        def handle_image_click(img, evt: gr.SelectData):
            global drawn_masks_per_id

            points_coords = (evt.index[0], evt.index[1])

            # 1- Check if there's a mask in that position.
            for drawn_mask_id in drawn_masks_per_id.keys():
                mask_data = drawn_masks_per_id[drawn_mask_id]

                mask = mask_data["mask"]

                if mask[points_coords[1], points_coords[0]]:
                    # 1.a) There's a mask, so "deselect it"
                    del drawn_masks_per_id[drawn_mask_id]

                    colors_arr = [
                        mask["last_color"] for mask in drawn_masks_per_id.values()
                    ]
                    masks_arr = [mask["mask"] for mask in drawn_masks_per_id.values()]

                    # Pass original image so the masks are refreshed
                    return masks_fn.get_masks_img(
                        original_image, masks_arr, colors_arr=colors_arr
                    )

            # 1.b) There's no mask, so create it. It will be added to the current segmented image
            # (it wont use the original image)
            masks = segment_ONNX_point(img, img_embedding, points_coords)

            # Convert (1, H, W) to (H, W)
            reshaped_masks = [np.reshape(m, m.shape[1:]) for m in masks]

            new_mask_color = masks_fn.get_random_color()

            masked_image = masks_fn.get_masks_img(
                img, reshaped_masks, color=new_mask_color
            )

            for mask in reshaped_masks:
                new_id = masks_fn.generate_mask_id()

                # Initialize the dict value. The last color property is set to a random color when adding
                drawn_masks_per_id[new_id] = {
                    "mask": mask,
                    "last_color": new_mask_color,
                }

            return masked_image

        ## Functions
        def start_segmentation(input):
            global segmented_masks
            global image_segmentated

            if input is None:
                logger.warning("No input loaded. Please, select an input image.")
                return

            input_image = input["image"]  # Shape: (H, W, 3)
            input_mask = input["mask"]  # Shape: (H, W, 4)

            # Reset variables
            segmented_masks = []

            # Convert to grayscale and obtain bounding boxes
            input_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)

            input_boxes = masks_fn.mask_bounding_boxes(input_mask)

            if len(input_boxes) == 0:
                segmented_masks = segment_automatic(input_image)
            else:
                segmented_masks = segment_boxes(
                    input_image, input_boxes, multimask_output
                )

            masked_image = masks_fn.get_masks_img(
                input_image, segmented_masks, random_colors=True
            )

            # Set the global holder for the segmentated image
            image_segmentated = masked_image

            # Image with the bounding box
            box_img = input_image.copy()
            for b in input_boxes:
                cv2.rectangle(box_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)

            logger.info("Segmentation completed.")
            return masked_image, box_img

        def start_inpainting(original_img, prompt_text, prompt_negative):
            global segmented_masks
            global selected_mask_indices

            input_img = original_img

            resized_img = cv2.resize(input_img, (512, 512))
            logger.info(
                f"Inpainting. Resizing from {input_img.shape} to {resized_img.shape}"
            )

            masks = None

            if current_tab == TABS.POINT:
                masks = [m["mask"] for m in drawn_masks_per_id.values()]
            elif current_tab == TABS.MASK:
                masks = [
                    value
                    for index, value in enumerate(segmented_masks)
                    if index in selected_mask_indices
                ]

            if len(masks) == 0:
                logger.warning("No masks selected. Please segment the image.")
                return

            logger.info("Joining masks...")
            joined_masks = masks_fn.join_masks_img(masks)

            # Convert to PIL Image the BGR cv2 image (the output will be RGB as PIL Image uses)
            PIL_img = img_fn.convert_from_cv2_to_image(
                cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            )
            b64_img = img_fn.img2b64(PIL_img)

            logger.info("Preprocessing image...")
            mask_preprocessed = masks_fn.preprocess_mask(
                joined_masks, (resized_img.shape[1], resized_img.shape[0])
            )
            b64_mask = img_fn.MAT2b64(mask_preprocessed)

            config = {
                Setting.Steps.value: UIConfig.get_cfg(Setting.Steps),
                Setting.InpaintingFill.value: UIConfig.get_cfg(Setting.InpaintingFill),
                Setting.DenoisingStrength.value: UIConfig.get_cfg(
                    Setting.DenoisingStrength
                ),
                Setting.DenoisingStrength.value: UIConfig.get_cfg(
                    Setting.DenoisingStrength
                ),
                Setting.ResizeMode.value: UIConfig.get_cfg(Setting.ResizeMode),
                Setting.ResizeMode.value: UIConfig.get_cfg(Setting.ResizeMode),
                Setting.InpaintFullResPadding.value: UIConfig.get_cfg(
                    Setting.InpaintFullResPadding
                ),
                "width": int(resized_img.shape[1]),
                "height": int(resized_img.shape[0]),
                "save_images": UIConfig.get_cfg(Setting.SaveOutput),
                "include_init_images": UIConfig.get_cfg(Setting.SaveOutput),
            }

            logger.info("Inpainting image...")
            inpaint_res = inpaint(
                prompt_text, prompt_negative, b64_img, b64_mask, config
            )
            if inpaint_res is None:
                return

            logger.info("Inpainting completed. ")
            image_result = img_fn.b642MAT(inpaint_res["imgb64"])

            image_id = inpaint_res["seed"]

            resized_result = cv2.resize(image_result, np.flip(original_image.shape[:2]))
            upscaled_result = None

            if UIConfig.get_cfg(Setting.SaveParams):
                save_data = {"prompt": prompt_text, "prompt_negative": prompt_negative}
                UIConfig.save_params(save_data)

            if UIConfig.get_cfg(Setting.SaveOutput):
                logger.info("Saving images")
                image_result_write = cv2.cvtColor(resized_result, cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"{out_path}/{image_id}_inpaint.png", image_result_write)
                cv2.imwrite(f"{out_path}/{image_id}_mask.png", mask_preprocessed)

            if UIConfig.get_cfg(Setting.UpscaleOutput):
                # upscale_to = (1024, 1280)
                upscaled_result = upscale_img(
                    img_fn.MAT2b64(resized_result),
                    # {
                    #     "resize_mode": 1,
                    #     "upscaling_resize_w": upscale_to[0],
                    #     "upscaling_resize_h": upscale_to[1],
                    #     "upscaling_crop": False,
                    # },
                )

                upscaled_result = cv2.cvtColor(upscaled_result, cv2.COLOR_BGR2RGB)

                if UIConfig.get_cfg(Setting.SaveOutput):
                    upscaled_result_write = cv2.cvtColor(
                        upscaled_result, cv2.COLOR_BGR2RGB
                    )

                    cv2.imwrite(
                        f"{out_path}/{image_id}_upscaled.png", upscaled_result_write
                    )

                logger.info("Saving and finishing upscaled image")

                return upscaled_result

            return resized_result

        def inpainting_generator(prompt_text, prompt_negative):
            operation = thread_fn.ThreadOperation(
                lambda: start_inpainting(original_image, prompt_text, prompt_negative)
            )
            operation.start()

            last_progress_img = None
            # Loop and check progress periodically.
            while True:
                try:
                    progress_image = get_sd_progress()
                except:
                    logger.exception(
                        "Error trying to get sd progress. Finishing generator."
                    )
                    break

                if operation.error:
                    logger.warning(
                        "Finishing generator due to an error inside the thread..."
                    )
                    yield last_progress_img
                    break
                if operation.result is not None:
                    logger.info("Finishing operation...")
                    yield operation.result
                    break
                else:
                    if progress_image is not None and not np.array_equal(
                        progress_image, last_progress_img
                    ):
                        # New image received
                        last_progress_img = progress_image
                        yield progress_image
                    elif last_progress_img:
                        yield last_progress_img

                time.sleep(1.5)

        # UI Movements
        points_tab.select(lambda: set_current_tab(TABS.POINT))
        masks_tab.select(lambda: set_current_tab(TABS.MASK))

        def handle_multimask_out_change(newValue):
            global multimask_output

            multimask_output = newValue

        multimask_output_check.change(
            handle_multimask_out_change,
            multimask_output_check,
        )

        image_masker.upload(
            gradio_fn.upload_img_callback(handle_image_upload, True),
            image_masker,
            image_masker,
            preprocess=False,
        )

        # On "Segment" button click
        segment_button.click(
            start_segmentation,
            inputs=image_masker,
            outputs=[segmented_image, segmented_boxs_image],
        )

        # On Mask selection in Segmented Image
        segmented_image.select(
            handle_mask_selection,
            outputs=segmented_image,
        )

        image_clicker.upload(
            gradio_fn.upload_img_callback(handle_image_upload, False),
            image_clicker,
            image_clicker,
            preprocess=False,
        )

        # On mask click
        image_clicker.select(
            handle_image_click,
            inputs=image_clicker,
            outputs=image_clicker,
            show_progress=False,
        )

        # inpaint_button.click(
        #     inpainting_generator,
        #     [prompt_text, prompt_negative],
        #     result_image,
        # )

        # prompt_text.submit(
        #     inpainting_generator, [prompt_text, prompt_negative], result_image
        # )

        # send_out_to_in_button.click(
        #     handle_send_to_input,
        #     inputs=result_image,
        #     outputs=[image_clicker, image_masker],
        # )

        # get_img_from_inpaint.click(
        #     handle_get_from_inpaint,
        #     outputs=image_clicker,
        #     _js="() => getImgFromInpaintTab()",
        # )

        return [
            demo,
            [
                image_clicker,
                multimask_output_check,
                image_masker,
                segment_button,
                segmented_image,
                segmented_boxs_image,
                # *config_elements,
                # get_img_from_inpaint,
            ],
        ]


if __name__ == "__main__":
    demo, _ = create_ui()

    # Queue enabled to use generators
    demo.queue()

    demo.launch(server_port=7861)
