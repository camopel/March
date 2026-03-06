"""March Plugin System — hooks, base class, and manager."""

from march.plugins._hooks import Hook
from march.plugins._base import Plugin
from march.plugins._manager import PluginManager

__all__ = ["Hook", "Plugin", "PluginManager"]
