import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
# first taken from https://medium.com/@lifanov.a.v/integrating-langgraph-with-ollama-for-advanced-llm-applications-d6c10262dafa
load_dotenv()

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time in the format %Y-%m-%d %H:%M:%S
    """
    return datetime.now().strftime(format)


tools = [get_now]

tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

model = ChatOllama(model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M")

#### alternative models that perform sufficiently well (miqu) or are the best open src alternative to the nous-hermes-2-mistral dpo finetune
# model = ChatOllama(model="mgmacleod/miqu-70b-q2", temp = 0,
# model = ChatOllama(model="eramax/senku", temp=0)
# model = ChatOllama(model="qwen:72b-text-v1.5-q2_K", temp=0.01)

###function calling only models that need prompts rewritten
# model = ChatOllama(model="gorilla-openfunctions-v2-q6_K", temp = 0)
# model = ChatOllama(model="joefamous/firefunction-v1:q3_k", temp=0.0)
# model = ChatOllama(model="openhermes")

### Alternative models that don't work currently 
# models that should work better
# model = ChatOllama(model="eas/nous-capybara:34b", temp=0)
# model = ChatOllama(model="starcoder2:15b",  temp = 0.0)
# model = ChatOllama(model="nous-hermes2:34b-yi-q5_K_M" , temp=

### coding models that either need better settings or don't do function calling well 
# model = ChatOllama(model="eramax/opencodeinterpreter:ds-33b-q4", temp=0)
# model = ChatOllama(model="dolphincoder")
# model = ChatOllama(model="dolphin-mixtral:8x7b-v2.7-q3_K_S", temp=0)
# model = ChatOllama(model="deepseek-coder:33b")
# model = ChatOllama(model="codellama:70b-instruct-q2_K", temp=0.0) #doesnt run


prompt = hub.pull("hwchase17/react")


agent_runnable = create_react_agent(model, tools, prompt)

def execute_tools(state):
    print("Called `execute_tools`")
    messages = [state["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state["agent_outcome"], response)]}


def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)


workflow.add_edge("action", "agent")
app = workflow.compile()

input_text = "Whats the current time? Provide just the time as the answer in the format: '%Y-%m-%d %H:%M:%S' without quotes ?"

inputs = {"input": input_text, "chat_history": []}

results = []
for s in app.stream(inputs):
    result = list(s.values())[0]
    results.append(result)
    print(result)

print(results[-1])