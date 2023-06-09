import logging
import numpy as np
import cv2
from skimage.measure import label, regionprops
from unique_id import get_unique_id

logger = logging.getLogger(__name__)


def get_center_mask(mask):
    # Flatten the mask to a 1-dimensional array
    flattened = mask.flatten()

    # Find the indices where the mask values are True
    indices = np.where(flattened)

    # Reshape the indices to match the original mask shape
    y_indices, x_indices = np.unravel_index(indices, mask.shape[:2])

    # Obtain the center coordinates
    center_y = np.mean(y_indices)
    center_x = np.mean(x_indices)

    return (int(center_y), int(center_x))


def get_masks_img(
    img, masks, color=[255, 0, 0], alpha=0.6, colors_arr=None, random_colors=False
):
    out_img = img.copy()

    for index, mask in enumerate(masks):
        if colors_arr is not None:
            if len(colors_arr) != len(masks):
                logger.error(
                    "Error: Color array map is wrong. The length of masks and colors_arr are different."
                )
                return

            overlay_color = colors_arr[index]
        elif random_colors:
            overlay_color = get_random_color()
        else:
            overlay_color = color

        # Copy of mask to do shape manipulations
        reshaped = mask

        if len(mask.shape) > 3:
            # (Y, W, 1, 1) case
            reshaped = np.max(mask, axis=-1)

        if len(mask.shape) == 2:
            # Greyscale image (W, H)
            reshaped = np.expand_dims(reshaped, 2)

        out_img = np.where(reshaped > 0, overlay_color * alpha, out_img)

    return out_img.astype(int)


def join_masks_img(masks):
    joined_masks = np.zeros(masks[0].shape)

    # Merging selecte dmasks
    for mask in masks:
        joined_masks += mask

    return joined_masks


def preprocess_mask(mask_array, shape):
    # Ensure mask array is of the correct shape and type
    mask_array = mask_array.astype(np.uint8)

    # Resize the mask array if needed
    mask_array = cv2.resize(mask_array, shape)

    # Convert mask to black and white
    _, mask_array = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY)

    return mask_array


def mask_bounding_boxes(mask):
    binary_image = np.where(mask > 0, 1, 0)

    labels = label(binary_image)

    props = regionprops(labels)

    bboxes = []

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox

        bbox = np.array([min_col, min_row, max_col, max_row])
        bboxes.append(bbox)

    return np.array(bboxes)


def generate_mask_id():
    return get_unique_id(length=8)


def get_random_color():
    return np.array(np.random.choice(range(256), size=3))


# Returns the nearest mask index (to the center) that belongs to a X, Y coordinates
def get_nearest_mask_index(masks, x, y):
    results = []

    for index, mask in enumerate(masks):
        # TODO: Handle case where there's no mask in these X Y corrdinates.
        if mask[y, x]:
            mask_center = get_center_mask(mask)
            distance_to_center = np.linalg.norm(np.array((y, x)) - mask_center)

            results.append({"distanceToCenter": distance_to_center, "index": index})

    if len(results) > 0:
        return sorted(results, key=lambda x: x["distanceToCenter"])[0]

    return None
