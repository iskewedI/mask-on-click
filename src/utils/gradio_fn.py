from src.utils.img_fn import convert_from_image_to_cv2
from gradio.processing_utils import decode_base64_to_image

# Fix for upload gradio method. The "upload" method is being called multiple times when
# data is uploaded, and the first times the data is null, and the internal decoding method
# breaks, making it impossible to use this event. With this fix, we are decoding it manually,
# *only if the DATA argument has a truthy value*
# The is_mask flag defines if the method is being called from a MASK image or just a IMAGE
# gradio component (in the first case, it will be a dict[image, mask], and the second just an image input)
# THIS NEEDS TO BE COMBINED WITH "preprocess: False" parameter to the upload function!!


def upload_img_callback(callback, is_mask=True):
    return lambda data: data and callback(
        convert_from_image_to_cv2(
            decode_base64_to_image(is_mask and data["image"] or data)
        )
    )
