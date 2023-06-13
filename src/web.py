import logging
import numpy as np
import cv2
import gradio as gr
from src.utils import gradio_fn, masks_fn, img_fn
from src.utils.static import TABS
from src.segmentator import (
    gen_image_embedding,
    segment_ONNX_point,
    segment_automatic,
    segment_boxes,
)
import os
from src.utils.logging import config_logs
import modules.scripts as scripts

scripts_basedir = scripts.basedir()

tracelogs_path = os.path.join(scripts_basedir, "trace.log")
config_logs(tracelogs_path)

logger = logging.getLogger(__name__)


# Configuration
styles_css = os.path.join(scripts_basedir, "styles.css")

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


def create_ui(opts):
    with gr.Blocks(css=styles_css):
        # with gr.Row():
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

        # Handles when the user clicks on the image on the click segmentation tab
        def segment_point_click(img, evt: gr.SelectData):
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
            return masked_image

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
            outputs=segmented_image,
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
            segment_point_click,
            inputs=image_clicker,
            outputs=image_clicker,
            show_progress=False,
        )

        return (
            image_clicker,
            multimask_output_check,
            image_masker,
            segment_button,
            segmented_image,
        )
