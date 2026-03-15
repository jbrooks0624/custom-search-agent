from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _fix_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively fix schema for OpenAI's strict structured output requirements:
    - Add additionalProperties: false to all object schemas
    - Ensure all properties are in the required array
    - Remove default values (not supported in strict mode)
    """
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False

        # Ensure all properties are required (OpenAI strict mode requirement)
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

            # Remove defaults from properties (not supported in strict mode)
            for prop_schema in schema["properties"].values():
                prop_schema.pop("default", None)
                _fix_schema_for_openai(prop_schema)

    if "items" in schema:
        _fix_schema_for_openai(schema["items"])

    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            _fix_schema_for_openai(def_schema)

    if "anyOf" in schema:
        for sub_schema in schema["anyOf"]:
            _fix_schema_for_openai(sub_schema)

    if "allOf" in schema:
        for sub_schema in schema["allOf"]:
            _fix_schema_for_openai(sub_schema)

    return schema


def create_response_format(model: type[T]) -> dict:
    """
    Create a structured output format specification for the OpenAI Responses API.

    This converts a Pydantic model into the format expected by OpenAI's
    structured output feature.

    Args:
        model: Pydantic model class defining the expected response structure

    Returns:
        Dictionary configuration for the 'text' parameter in responses.create()
    """
    schema = model.model_json_schema()
    schema = _fix_schema_for_openai(schema)

    return {
        "format": {
            "type": "json_schema",
            "name": model.__name__,
            "schema": schema,
            "strict": True,
        }
    }


def validate_response(content: str, model: type[T]) -> T:
    """
    Validate and parse a JSON response into a Pydantic model.

    Args:
        content: JSON string from the API response
        model: Pydantic model class to parse into

    Returns:
        Parsed and validated Pydantic model instance

    Raises:
        ValidationError: If the content doesn't match the model schema
    """
    return model.model_validate_json(content)
