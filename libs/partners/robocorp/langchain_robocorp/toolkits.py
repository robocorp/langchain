"""Robocorp Action Server toolkit."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List
from urllib.parse import urljoin

import requests
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tracers.context import _tracing_v2_is_enabled
from langsmith import Client

from langchain_robocorp._common import (
    get_required_param_descriptions,
    reduce_openapi_spec,
)
from langchain_robocorp._prompts import (
    TOOLKIT_TOOL_DESCRIPTION,
)

MAX_RESPONSE_LENGTH = 5000
LLM_TRACE_HEADER = "X-action-trace"


class RunDetailsCallbackHandler(BaseCallbackHandler):
    def __init__(self, run_details: dict) -> None:
        self.run_details = run_details

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        if "parent_run_id" in kwargs:
            self.run_details["run_id"] = kwargs["parent_run_id"]
        else:
            if "run_id" in self.run_details:
                self.run_details.pop("run_id")


def get_param_details(endpoint_spec: dict) -> list:
    """Get an OpenAPI endpoint parameter details"""
    param_details = []

    schema = (
        endpoint_spec.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for key, value in properties.items():
        detail = {
            "name": key,
            "type": value.get("type", "string"),  # Default to string if unspecified
            "description": value.get("description", ""),
            "required": key in required_fields,
            # Add other necessary details like default values, etc.
        }
        param_details.append(detail)

    return param_details


type_mapping = {
    "string": str,  # For JSON strings
    "integer": int,  # For JSON numbers (integers)
    "number": float,  # For JSON numbers (floats)
    "object": dict,  # For JSON objects
    "array": list,  # For JSON arrays
    "boolean": bool,  # For JSON booleans
    "null": type(None),  # For JSON null
}


class ActionServerToolkit(BaseModel):
    """Toolkit exposing Robocorp Action Server provided actions as individual tools."""

    url: str = Field(exclude=True)
    """Action Server URL"""
    llm: BaseChatModel
    """Chat model to be used for the Toolkit agent"""
    api_key: str = Field(exclude=True, default="")
    """Action Server request API key"""
    report_trace: bool = Field(exclude=True, default=False)
    """Enable reporting Langsmith trace to Action Server runs"""

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self, **kwargs: Any) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        # Fetch and format the API spec
        try:
            spec_url = urljoin(self.url, "openapi.json")
            response = requests.get(spec_url)
            json_spec = response.json()
            api_spec = reduce_openapi_spec(self.url, json_spec)
        except Exception:
            raise ValueError(
                f"Failed to fetch OpenAPI schema from Action Server - {self.url}"
            )

        # Prepare request tools
        run_details: dict = {}

        # Prepare callback manager
        callback_manager = kwargs.get("callback_manager", CallbackManager([]))
        callbacks: List[BaseCallbackHandler] = []

        if _tracing_v2_is_enabled():
            callbacks.append(RunDetailsCallbackHandler(run_details))

        for callback in callbacks:
            callback_manager.add_handler(callback)

        # Prepare the toolkit
        toolkit: List[BaseTool] = []

        prompt_variables = {
            "api_url": self.url,
        }

        def create_function(endpoint: str) -> Callable:
            def func(**data: Any) -> Any:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    if self.report_trace and "run_id" in run_details:
                        client = Client()
                        run = client.read_run(self.run_details["run_id"])
                        if run.url:
                            headers[LLM_TRACE_HEADER] = run.url
                except Exception:
                    pass

                response = requests.post(
                    endpoint, headers=headers, data=json.dumps(data)
                )
                output = response.text[:MAX_RESPONSE_LENGTH]

                return output

            return func

        for endpoint_name, docs in api_spec.endpoints:
            if not endpoint_name.startswith("/api/actions"):
                continue

            tool_name = f"robocorp_action_server_{docs['operationId']}"
            summary = docs["summary"]
            tool_description = TOOLKIT_TOOL_DESCRIPTION.format(
                name=summary,
                description=docs.get("description", summary),
                required_params=get_required_param_descriptions(docs),
            )

            prompt_variables[
                "api_docs"
            ] = f"{endpoint_name}: \n{json.dumps(docs, indent=4)}"

            param_details = get_param_details(docs)
            fields = {}
            for param in param_details:
                fields[param["name"]] = (type_mapping[param["type"]], ...)
            schema_model = create_model("DynamicToolInputSchema", **fields)  # type: ignore

            toolkit.append(
                StructuredTool(
                    name=tool_name,
                    func=create_function(urljoin(self.url, endpoint_name)),
                    description=tool_description,
                    args_schema=schema_model,
                    callback_manager=callback_manager,
                )
            )

        return toolkit
