from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import RobocorpToolkit
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize Robocorp Toolkit
toolkit = RobocorpToolkit(url="http://localhost:8080")
tools = toolkit.get_tools(llm=llm)

# Initialize Agent
template = """You are a helpful assistant"""
prompt = ChatPromptTemplate.from_messages(
  [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = OpenAIFunctionsAgent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Typings for Langserve playground
class AgentInputs(BaseModel):
    input: str

agent_executor = agent_executor.with_types(input_type=AgentInputs)
