"""Robocorp Action Server toolkit."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
import json

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain.tools.requests.tool import BaseRequestsTool
from langchain.agents.agent_toolkits.robocorp.prompts import (
    API_CONTROLLER_PROMPT,
    REQUESTS_POST_TOOL_DESCRIPTION,
    REQUESTS_RESPONSE_PROMPT,
    TOOLKIT_TOOL_DESCRIPTION,
)
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.agents.agent_toolkits.robocorp.spec import (
    reduce_openapi_spec,
    get_required_param_descriptions,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities.requests import RequestsWrapper
from langchain.prompts import PromptTemplate
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langsmith import Client


MAX_RESPONSE_LENGTH = 5000
LLM_TRACE_HEADER = "X-llm-trace"


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


class ToolInputSchema(BaseModel):
    question: str = Field(...)


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests POST tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_post"
    """Tool name."""
    description = REQUESTS_POST_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain
    run_details: dict

    def _run(self, text: str, **kwargs: Any) -> str:
        try:
            data = json.loads(text)

        except json.JSONDecodeError as e:
            raise e

        try:
            if "run_id" in self.run_details:
                client = Client()
                run = client.read_run(self.run_details["run_id"])
                self.requests_wrapper.headers[LLM_TRACE_HEADER] = run.url
        except Exception:
            if LLM_TRACE_HEADER in self.requests_wrapper.headers:
                self.requests_wrapper.headers.pop(LLM_TRACE_HEADER)

        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]

        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RobocorpToolkit(BaseToolkit):
    """Toolkit exposing Robocorp Action Server provided actions."""

    url: str = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        # Fetch and format the API spec
        response = requests.get(f"{self.url}/openapi.yaml")
        api_spec = reduce_openapi_spec(response.text)

        # Prepare request tools
        llm_chain = LLMChain(llm=self.llm, prompt=REQUESTS_RESPONSE_PROMPT)

        requests_wrapper = RequestsWrapper(headers={})
        run_details = {}

        tools: List[BaseTool] = [
            RequestsPostToolWithParsing(
                requests_wrapper=requests_wrapper,
                llm_chain=llm_chain,
                run_details=run_details,
            ),
        ]

        tool_names = ", ".join([tool.name for tool in tools])
        tool_descriptions = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )

        toolkit: List[BaseTool] = []

        prompt_variables = {
            "api_url": self.url,
            "tool_names": tool_names,
            "tool_descriptions": tool_descriptions,
        }

        callbacks: List[BaseCallbackHandler] = []

        with tracing_v2_enabled():
            callbacks.append(RunDetailsCallbackHandler(run_details))

        callback_manager = CallbackManager(callbacks)

        # Prepare the toolkit
        for name, _, docs in api_spec.endpoints:
            if not name.startswith("/api/actions"):
                continue

            tool_name = f"robocorp_action_server_{docs['operationId']}"
            tool_description = TOOLKIT_TOOL_DESCRIPTION.format(
                name=docs["summary"],
                description=docs["description"],
                required_params=get_required_param_descriptions(docs),
            )

            prompt_variables["api_docs"] = f"{name}: \n{json.dumps(docs, indent=4)}"
            prompt = PromptTemplate(
                template=API_CONTROLLER_PROMPT,
                input_variables=["input", "agent_scratchpad"],
                partial_variables=prompt_variables,
            )

            agent = ZeroShotAgent(
                llm_chain=LLMChain(llm=self.llm, prompt=prompt),
                allowed_tools=[tool.name for tool in tools],
            )

            executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=True,
                tags=["robocorp-action-server"],
            )

            toolkit.append(
                Tool(
                    name=tool_name,
                    func=executor.run,
                    description=tool_description,
                    args_schema=ToolInputSchema,
                    callback_manager=callback_manager,
                )
            )

        return toolkit
