"""Configuration utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
