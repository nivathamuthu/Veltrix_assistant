"""A small JSON validator for the model outputs we ask for.

The LLM's JSON is untrusted: it is parsed and checked against a schema before any code
uses it. Only the JSON-Schema subset the project's schemas use is supported (``type`` as a
name or a list of names, ``enum``, ``properties`` / ``required`` / ``additionalProperties:
false``, ``items``, ``maxLength``, ``maxItems``). An unknown keyword is a programming error
and raises, so a schema can never silently validate less than it appears to.

``parse_json_object`` tolerates the one habit models have that is harmless to undo: a
```json fenced block around the object. Anything else that is not one JSON value is invalid.
"""
from __future__ import annotations

import json
import re
from typing import Any, Final

_FENCE: Final = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_KNOWN: Final = frozenset(
    {"type", "enum", "properties", "required", "additionalProperties", "items", "maxLength",
     "maxItems", "title", "description"}
)


class InvalidModelOutput(ValueError):
    """The model's text is not JSON, or does not match the schema. Carries no model text."""


def parse_json_object(text: object) -> Any:
    if not isinstance(text, str):
        raise InvalidModelOutput("response is not text")
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise InvalidModelOutput("response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise InvalidModelOutput("response is not a JSON object")
    return value


def _type_ok(value: Any, name: str) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise ValueError(f"unsupported schema type {name!r}")


def _check(value: Any, schema: dict, path: str) -> None:
    unknown = set(schema) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
    wanted = schema.get("type")
    if wanted is not None:
        names = [wanted] if isinstance(wanted, str) else list(wanted)
        if not any(_type_ok(value, n) for n in names):
            raise InvalidModelOutput(f"{path or 'value'}: wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidModelOutput(f"{path or 'value'}: not an allowed value")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        raise InvalidModelOutput(f"{path or 'value'}: too long")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise InvalidModelOutput(f"{path or 'value'}: too many items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(value):
                _check(item, item_schema, f"{path}[{i}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", ()):
            if name not in value:
                raise InvalidModelOutput(f"{path}.{name}: missing")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise InvalidModelOutput(f"{path or 'value'}: unexpected fields")
        for name, sub in properties.items():
            if name in value:
                _check(value[name], sub, f"{path}.{name}")


def validate(value: Any, schema: dict) -> Any:
    """Return ``value`` if it matches ``schema``; raise ``InvalidModelOutput`` otherwise."""
    _check(value, schema, "")
    return value


def parse_and_validate(text: object, schema: dict) -> Any:
    return validate(parse_json_object(text), schema)
