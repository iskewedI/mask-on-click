import base64
import cv2
import numpy as np
from PIL import Image
import io


# B64 transformations
def MAT2b64(mat):
    return base64.b64encode(cv2.imencode(".jpg", mat)[1]).decode()


def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")


def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return _b64encode(buffered.getvalue())


def b642img(b64str) -> Image:
    imgdata = base64.b64decode(b64str.split(",", 1)[0])
    return Image.open(io.BytesIO(imgdata))


def b642MAT(b64str) -> np.ndarray:
    imgbytes = base64.b64decode(b64str)
    return np.array(Image.open(io.BytesIO(imgbytes)))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def is_similar(image1, image2):
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())
