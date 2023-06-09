import logging
import gradio as gr
import os
import json
import modules.scripts as scripts

logger = logging.getLogger(__name__)

last_params = {}
params_file_path = os.path.join(scripts.basedir(), "last_params.json")


def load_params():
    global last_params

    if os.path.isfile(params_file_path):
        params_file = open(params_file_path)
        last_params = json.load(params_file)
        params_file.close()


def get_param(name):
    return last_params.get(name)


def save_params(data):
    params = last_params.copy()

    params.update(data)

    serialized = json.dumps(params, indent=4)

    with open(params_file_path, "w") as outfile:
        outfile.write(serialized)
        outfile.close()


def get_cfg(name):
    result = [
        obj["value"] for sublist in config for obj in sublist if obj.get("name") == name
    ]

    if len(result) > 0:
        return result[0]

    return None


# Each array is a new row
config = [[]]


def change_cfg(name, rowIndex, value):
    logger.debug(f"changing cfg {name} to {value}")
    for cfg in config[rowIndex]:
        if cfg["name"] == name:
            cfg["value"] = value

    return value


def map_cfg_to_component(index, rowIndex, cfg):
    extra_params = {}

    if cfg.get("params"):
        extra_params = cfg["params"]

    element = cfg["component"](value=cfg["value"], label=cfg["label"], **extra_params)

    element.change(
        lambda v: change_cfg(config[rowIndex][index]["name"], rowIndex, v),
        element,
        element,
    )

    return element


def get_ui():
    config_elements = []

    for rowIndex, _ in enumerate(config):
        with gr.Row(variant="compact"):
            # Helper function to map a config value to a Checkbox element

            for index, cfg in enumerate(config[rowIndex]):
                el = map_cfg_to_component(index, rowIndex, cfg)

                config_elements.append(el)

    return config_elements
