import os

import yaml

from dagnostics.core.models import AppConfig


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Loads configuration from a YAML file and validates it against AppConfig Pydantic model.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        raw_config_dict = yaml.safe_load(f)

    config = AppConfig(**raw_config_dict)
    return config
