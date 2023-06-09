import logging
import os
import torch
import onnxruntime
import numpy as np
from memoization import cached
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from src.utils.perf_fn import measure_performance
import modules.scripts as scripts

logger = logging.getLogger(__name__)

sam_checkpoint = os.path.join(scripts.basedir(), "pretrained", "sam_model.pth")
quantized_onnx_model = os.path.join(
    scripts.basedir(), "pretrained", "model_quantized.onnx"
)

model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

logger.info("Loading model...")
sam.to(device=device)

logger.info("Instantiating generator...")
automatic_mask_generator = SamAutomaticMaskGenerator(sam)
sam_predictor = SamPredictor(sam)

logger.info("Loading ORT session")
ort_session = onnxruntime.InferenceSession(quantized_onnx_model)


def gen_image_embedding(img):
    logger.info("Generating image embedding")

    sam_predictor.set_image(img)

    return sam_predictor.get_image_embedding().cpu().numpy()


# return shape => [(X, Y)]
def segment_automatic(img):
    logger.info("Segmenting with automatic mask generator...")
    masks_generated = automatic_mask_generator.generate(img)

    return [mask["segmentation"] for mask in masks_generated]


def segment_boxes(img, boxes, multimask=False) -> torch.Tensor:
    logger.info("Segmenting with box segmentation...")

    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        input_boxes, img.shape[:2]
    )

    logger.info("Segmenting image with SamPredictor...")
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=multimask,
    )

    # torch.Size([InputMaskAmount, ResultMaskAmount, X, Y])
    result_masks = masks

    logger.info("Segmentation shape => ", masks.shape)

    if not multimask:
        # to (288, 576, 2)
        stacked_masks = np.stack(
            [mask_tensor.cpu().squeeze() for mask_tensor in masks], axis=-1
        )

        if stacked_masks.shape[2] > 1:
            # Split over the third dimension, so we get all the maskes separated in an array.
            # (288, 576, 2) TO [(288, 576, 1)]
            result_masks = np.dsplit(stacked_masks, 2)
        else:
            # If there's just one mask, convert it to a list[ndarr] to use consistent values
            result_masks = [stacked_masks]
    else:
        output_masks = []  # List to store all the output masks

        # Iterating over each mask
        for n in range(result_masks.shape[0]):
            # Iterating over each channel
            for c in range(result_masks.shape[1]):
                mask = result_masks[
                    n, c, :, :
                ]  # This gives a 2D mask of shape (288, 576)
                mask = mask.unsqueeze(
                    -1
                )  # Expanding dimensions to get shape (288, 576, 1)
                mask_np = (
                    mask.detach().cpu().numpy()
                )  # Convert the tensor to a numpy array
                output_masks.append(mask_np)

        result_masks = output_masks

    logger.info("Result mask shape => ", result_masks[0].shape)
    return result_masks


@measure_performance
@cached
def segment_ONNX_point(image, embedding, point_cords):
    if embedding is None:
        logger.error("Image embedding not received. Can't segment.")
        return

    onnx_coord = torch.tensor([[point_cords, [0.0, 0.0]]], dtype=torch.float32)
    onnx_label = torch.tensor([[1, -1]], dtype=torch.float32)

    onnx_coord = sam_predictor.transform.apply_coords(
        onnx_coord.numpy(), image.shape[:2]
    ).astype(np.float32)

    onnx_mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    onnx_has_mask_input = torch.zeros(1, dtype=torch.float32)

    ort_inputs = {
        "image_embeddings": embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label.numpy(),
        "mask_input": onnx_mask_input.numpy(),
        "has_mask_input": onnx_has_mask_input.numpy(),
        "orig_im_size": torch.tensor(image.shape[:2], dtype=torch.float32).numpy(),
    }

    masks, _, _ = ort_session.run(None, ort_inputs)
    masks = masks > sam_predictor.model.mask_threshold

    return masks
