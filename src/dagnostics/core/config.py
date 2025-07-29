import os
from pathlib import Path
from typing import Optional

import yaml

from dagnostics.core.models import AppConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Loads configuration from a YAML file with multiple fallback options.

    Priority order:
    1. Explicit config_path parameter
    2. DAGNOSTICS_CONFIG environment variable
    3. ~/.dagnostics/config.yaml (user config)
    4. ./config.yaml (current directory)
    5. ./config/config.yaml (development structure)

    Args:
        config_path: Optional explicit path to config file

    Returns:
        AppConfig: Validated configuration object

    Raises:
        FileNotFoundError: If no valid config file is found
    """
    search_paths = []

    # 1. Explicit path takes highest priority
    if config_path:
        search_paths.append(config_path)

    # 2. Environment variable
    env_config = os.getenv("DAGNOSTICS_CONFIG")
    if env_config:
        search_paths.append(env_config)

    # 3. User home directory config
    home_config = Path.home() / ".dagnostics" / "config.yaml"
    search_paths.append(str(home_config))

    # 4. Current working directory
    search_paths.append("config.yaml")

    # 5. Development structure (for backwards compatibility)
    search_paths.append("config/config.yaml")

    # Try each path in order
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw_config_dict = yaml.safe_load(f)
                config = AppConfig(**raw_config_dict)
                print(f"Loaded configuration from: {path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
                continue

    # If no config found, provide helpful error message
    raise FileNotFoundError(
        "Configuration file not found. Searched in:\n"
        + "\n".join(f"  - {path}" for path in search_paths)
        + "\n\nCreate a config file at one of these locations or set DAGNOSTICS_CONFIG environment variable."
    )
