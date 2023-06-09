import launch
import os

segment_anything_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "segment-anything"
)

if not os.path.exists(segment_anything_path):
    launch.run(
        f"git clone https://github.com/facebookresearch/segment-anything {segment_anything_path}"
    )
    if not launch.is_installed("onnx"):
        launch.run(f"cd {segment_anything_path} && pip install -e .")

if not launch.is_installed("unique-id"):
    launch.run_pip(
        "install unique-id", "unique-id requirement for segmentation-on-click"
    )
if not launch.is_installed("memoization"):
    launch.run_pip(
        "install memoization", "memoization requirement for segmentation-on-click"
    )
if not launch.is_installed("decorator"):
    launch.run_pip(
        "install decorator", "decorator requirement for segmentation-on-click"
    )
