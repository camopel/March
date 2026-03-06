"""Environment variable interpolation for config values.

Supports ${VAR} and ${VAR:default} syntax inside string values.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Matches ${VAR} or ${VAR:default_value}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def interpolate_value(value: str) -> str:
    """Expand ${ENV_VAR} and ${ENV_VAR:default} in a single string value.

    Args:
        value: String that may contain ${VAR} or ${VAR:default} patterns.

    Returns:
        String with all patterns replaced by environment variable values or defaults.

    Raises:
        ValueError: If an environment variable is referenced without a default and is not set.
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable ${{{var_name}}} is not set and has no default"
        )

    return _ENV_PATTERN.sub(_replace, value)


def interpolate_config(data: Any) -> Any:
    """Recursively interpolate environment variables in a config data structure.

    Walks dicts, lists, and strings. Non-string leaf values are returned unchanged.

    Args:
        data: A config data structure (dict, list, str, int, float, bool, None).

    Returns:
        The same structure with all ${VAR} patterns in strings expanded.
    """
    if isinstance(data, dict):
        return {key: interpolate_config(val) for key, val in data.items()}
    if isinstance(data, list):
        return [interpolate_config(item) for item in data]
    if isinstance(data, str):
        return interpolate_value(data)
    return data
