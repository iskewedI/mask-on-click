# MaskOnClick: A Segment Anything Extension for Stable Diffusion Web UI

## Description
MaskOnClick is a powerful extension for Stable Diffusion Web UI, leveraging the potential of Meta's Segment Anything, an open source ONNX model. It provides an interactive tool for image segmentation, enabling users to swiftly generate masks within an image. Designed specifically for image editing and enhancing purposes, this extension allows the users to highlight and edit specific areas of an image.

This extension is extremely useful when combined with the inpainting technique provided by Stable Diffusion. For example, users can use it to replace the clothes of a person in an image or edit any specific object or area. Written in Python 3.10 and Torch 2.0, MaskOnClick offers a simple yet efficient solution for advanced image processing tasks.

## Installation
### Prerequisite:
Make sure you have Stable Diffusion Web UI installed on your system.

To install MaskOnClick, please follow these steps:

1. Clone the repository

```bash
git clone https://github.com/iskewedI/mask-on-click.git
```

2. Move into the directory
```bash
cd mask-on-click
```

3. Install the required packages
```bash
pip install -r requirements.txt
```
4. Install the extension to your Stable Diffusion Web UI project following its specific extension installation guidelines.
   You can also just clone the repo inside the Stable Diffusion WebUI extensions directory and reload the WebUI.

## Usage
After installing the extension, you can access the MaskOnClick functionality directly from the Stable Diffusion Web UI interface. Simply choose an image for editing, and you can start segmenting areas of the image with a simple click, after which you can proceed with the inpainting or any other editing operations as per your requirement.

## Built With
- Python 3.10
- PyTorch 2.0
- Gradio
- Meta's Segment Anything

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License
Distributed under the MIT License. See LICENSE for more information.

Please feel free to contact if you have any questions or need further information.

## Acknowledgements
- [Automatic1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Meta's Segment Anything](https://github.com/facebookresearch/segment-anything)
