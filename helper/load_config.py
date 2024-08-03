import json

from helper.constants import CONFIG_FILE_PATH, EMBEDDING_MODELS


def load_config():
    if CONFIG_FILE_PATH.exists():
        with open(CONFIG_FILE_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Set default values if keys are missing
    if "last_path" not in config:
        config["last_path"] = ""
    if "saved_paths" not in config:
        config["saved_paths"] = []
    if (
        "last_embedding_model" not in config
        or config["last_embedding_model"] not in EMBEDDING_MODELS
    ):
        config["last_embedding_model"] = list(EMBEDDING_MODELS.keys())[
            0
        ]  # Default to the first model

    return config


def save_config(config):
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f)
