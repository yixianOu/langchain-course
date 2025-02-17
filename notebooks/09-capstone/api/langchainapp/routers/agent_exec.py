import asyncio
import os

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Constants and Configuration
OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])

# Tools definition
@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

@tool
def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
    return {"answer": answer, "tools_used": tools_used}

tools = [add, subtract, multiply, exponentiate, final_answer]
name2tool = {tool.name: tool.func for tool in tools}

# LLM and Prompt Setup
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    streaming=True,
    api_key=OPENAI_API_KEY
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. You MUST "
        "then use the final_answer tool to provide a final answer to the user. "
        "DO NOT use the same tool more than once."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            if token_or_done:
                yield token_or_done
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

# Agent Executor
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            async def stream(query: str):
                response = self.agent.with_config(
                    callbacks=[streamer]
                )
                # we initialize the output dictionary that we will be populating with
                # our streamed output
                output = None
                # now we begin streaming
                async for token in response.astream({
                    "input": query,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad
                }):
                    if output is None:
                        output = token
                    else:
                        # we can just add the tokens together as they are streamed and
                        # we'll have the full response object at the end
                        output += token
                    if token.content != "":
                        # we can capture various parts of the response object
                        if verbose: print(f"content: {token.content}", flush=True)
                    tool_calls = token.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        if verbose: print(f"tool_calls: {tool_calls}", flush=True)
                        tool_name = tool_calls[0]["function"]["name"]
                        if tool_name:
                            if verbose: print(f"tool_name: {tool_name}", flush=True)
                        arg = tool_calls[0]["function"]["arguments"]
                        if arg != "":
                            if verbose: print(f"arg: {arg}", flush=True)
                return AIMessage(
                    content=output.content,
                    tool_calls=output.tool_calls,
                    tool_call_id=output.tool_calls[0]["id"]
                )

            tool_call = await stream(query=input)
            # add initial tool call to scratchpad
            agent_scratchpad.append(tool_call)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_call_id
            tool_out = name2tool[tool_name](**tool_args)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history, we only add the "answer" field
        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return tool_args

# Helper Functions
async def token_generator(query: str, streamer):
    task = asyncio.create_task(agent_executor.invoke(query, streamer))
    async for token in streamer:
        if token == "<<STEP_END>>":
            yield "\n\n"
        elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
            if tool_name := tool_calls[0]["function"]["name"]:
                yield tool_name
            if tool_args := tool_calls[0]["function"]["arguments"]:
                yield tool_args
    await task

# Initialize agent executor
agent_executor = CustomAgentExecutor()