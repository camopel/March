"""Tool system foundation for the March agent framework.

Provides the @tool decorator, Tool class, and ToolMeta for defining agent tools.
The decorator automatically extracts parameter schemas from Python type hints.
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, get_type_hints


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string."""
    # Handle Optional[X] (Union[X, None])
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        # Handle list[X], dict[X, Y]
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        # Handle Optional (Union with None)
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json_schema(non_none[0])

    return _TYPE_MAP.get(annotation, "string")


def _extract_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema parameters from a function's type hints and docstring.

    Inspects the function signature to build a JSON Schema 'object' type
    with properties derived from parameter names, type hints, and defaults.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Parse docstring for parameter descriptions
    param_docs = _parse_param_docs(fn.__doc__ or "")

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(annotation)
        prop: dict[str, Any] = {"type": json_type}

        # Add description from docstring
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]

        # Add default value
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        # Handle list item types
        if json_type == "array":
            args = getattr(annotation, "__args__", ())
            if args:
                item_type = _python_type_to_json_schema(args[0])
                prop["items"] = {"type": item_type}

        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Parse parameter descriptions from a docstring.

    Supports Google-style (Args:) and Sphinx-style (:param x:) docstrings.
    """
    result: dict[str, str] = {}
    if not docstring:
        return result

    lines = docstring.split("\n")
    in_args = False
    current_param: str | None = None

    for line in lines:
        stripped = line.strip()

        # Google-style: Args:
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith("-") and ":" not in stripped:
                # End of Args section
                if not stripped.startswith(" "):
                    in_args = False
                    continue
            # Parse "param_name: description" or "param_name (type): description"
            if ":" in stripped:
                parts = stripped.lstrip("- ").split(":", 1)
                param_part = parts[0].strip()
                # Remove type hints in parens
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                result[param_part] = desc
                current_param = param_part
            elif current_param and stripped:
                # Continuation of previous param description
                result[current_param] += " " + stripped

        # Sphinx-style: :param x: description
        if stripped.startswith(":param "):
            rest = stripped[7:]
            if ":" in rest:
                parts = rest.split(":", 1)
                param_name = parts[0].strip()
                desc = parts[1].strip()
                result[param_name] = desc

    return result


@dataclass
class ToolMeta:
    """Metadata for a tool, used for registration and LLM schema generation.

    Attributes:
        name: Tool name (used in tool_call.name).
        description: Human-readable description for the LLM.
        parameters: JSON Schema for the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A registered tool that can be executed by the agent.

    Attributes:
        name: Tool name.
        description: Description for the LLM.
        parameters: JSON Schema for parameters.
        fn: The async callable to execute.
        source: Where this tool comes from (e.g. "builtin", "mcp:server_name").
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Coroutine[Any, Any, str]]
    source: str = "builtin"

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for LLM tool_use.

        Returns the tool definition in the format expected by OpenAI/Anthropic APIs.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Decorator to register a function as an agent tool.

    Automatically extracts parameter schemas from Python type hints.

    Args:
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function's docstring.

    Usage:
        @tool(description="Read contents of a file")
        async def file_read(path: str, offset: int = 0) -> str:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        params = _extract_schema(fn)

        fn._tool_meta = ToolMeta(  # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc,
            parameters=params,
        )
        return fn

    return decorator
