from pathlib import Path

import yaml

from dagnostics.core.models import AppConfig


def load_config() -> AppConfig:
    """Load and parse the application configuration."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
            return AppConfig(**raw_config)
    else:
        raise FileNotFoundError("Configuration file not found: config/config.yaml")
