import requests
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

base_url = "http://127.0.0.1:7860"
api_v1_url = f"{base_url}/sdapi/v1"


def img2img(options):
    response = requests.post(url=f"{api_v1_url}/img2img", json=options)

    return response.json()


def interrogate(b64img, options):
    options["image"] = b64img

    response = requests.post(url=f"{api_v1_url}/interrogate", json=options)

    return response.json()


def upscale(b64img, options):
    options["image"] = b64img

    response = requests.post(url=f"{api_v1_url}/extra-single-image", json=options)

    return response.json()


def get_progress():
    response = requests.get(url=f"{api_v1_url}/progress")

    return response.json()
