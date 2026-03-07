"""Config loader — loads, interpolates, validates, and caches March configuration.

Loads from ~/.march/config.yaml. Fails fast if not initialized.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from march.config.interpolation import interpolate_config
from march.config.schema import MarchConfig

DEFAULT_CONFIG_DIR = Path.home() / ".march"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"

# Module-level cache for the loaded config
_cached_config: MarchConfig | None = None


class ConfigNotFoundError(Exception):
    """Raised when config.yaml doesn't exist."""
    pass


def ensure_config_exists(config_path: Path | None = None) -> Path:
    """Ensure the config file exists, or fail with init instructions.

    Args:
        config_path: Path to the config file. Defaults to ~/.march/config.yaml.

    Returns:
        The resolved path to the config file.

    Raises:
        ConfigNotFoundError: If the config file doesn't exist.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigNotFoundError(
            f"Config not found: {path}\n"
            f"Run 'march init' to set up March."
        )
    return path


def load_raw_yaml(config_path: Path) -> dict:
    """Load raw YAML data from a config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed YAML data as a dict. Returns empty dict for empty files.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    content = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data).__name__}")
    return data


def load_config(
    config_path: Path | None = None,
    *,
    use_cache: bool = True,
    interpolate: bool = True,
) -> MarchConfig:
    """Load, interpolate, and validate March configuration.

    This is the primary entry point for loading config. It:
    1. Ensures the config file exists (creates defaults if needed)
    2. Loads the raw YAML
    3. Interpolates ${ENV_VAR:default} patterns in string values
    4. Validates through Pydantic models

    Args:
        config_path: Path to config file. Defaults to ~/.march/config.yaml.
        use_cache: If True, return cached config on subsequent calls. Default True.
        interpolate: If True, expand ${ENV_VAR:default} patterns. Default True.

    Returns:
        Validated MarchConfig instance.

    Raises:
        pydantic.ValidationError: If config doesn't match the schema.
        ValueError: If a required env var is missing.
        yaml.YAMLError: If the YAML is malformed.
    """
    global _cached_config

    if use_cache and _cached_config is not None:
        return _cached_config

    path = ensure_config_exists(config_path)
    raw_data = load_raw_yaml(path)

    if interpolate:
        raw_data = interpolate_config(raw_data)

    config = MarchConfig.model_validate(raw_data)

    if use_cache:
        _cached_config = config

    return config


def reset_cache() -> None:
    """Clear the cached config. Useful for testing or config reload."""
    global _cached_config
    _cached_config = None


def get_config() -> MarchConfig:
    """Get the current cached config, loading it if needed.

    Convenience function for modules that need the config but don't want
    to pass it around.

    Returns:
        The current MarchConfig instance.
    """
    if _cached_config is not None:
        return _cached_config
    return load_config()
