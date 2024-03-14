from __future__ import annotations
from langchain_community.llms import Ollama
import math
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

import getpass
import os
import json
import yaml

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation


from langchain.chains import create_structured_output_runnable
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain as as_runnable

from collections import defaultdict, deque
from typing_extensions import TypedDict

# cation or a response.
from typing import List
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.runnables import RunnableConfig

# def _set_if_undefined(var: str) -> None:
#     if os.environ.get(var):
#         return
#     os.environ[var] = getpass.getpass(var)

# Optional: Configure tracing to visualize and debug the agent
# _set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ''
# os.environ["OPENAI_API_KEY"] = "ENTER API KEY FOR TESTING WITH  gpt-4-turbo-preview"

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M" #"dolphin-mixtral:8x7b-v2.7-q3_K_S" # crea"dolphincoder"# "eramax/opencodeinterpreter:ds-33b-q4"
os.environ["OPENAI_API_KEY"] =  "NA"

class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Select the child with the highest UCT to search next."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str

from langchain_openai import ChatOpenAI
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.chat_models import ChatOllama

# llm = ChatOpenAI(model="gpt-4-turbo-preview")
llm = ChatOpenAI(model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M")
# llm = OllamaFunctions(model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M")
# llm = ChatOllama(model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M")

#original notebook used tavily instead replaced with duckduckgo
# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
# tools = [tavily_tool]
from langchain_community.llms import VLLMOpenAI, VLLM

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    model_name="olafgeibig/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
    model_kwargs={"stop": ["."]},
)

llm = VLLM(model="NousResearch/Hermes-2-Pro-Mistral-7B",
        #    "olafgeibig/Nous-Hermes-2-Mistral-7B-DPO-GGUF/blob/main/nous-hermes-2-mistral-7b-dpo.Q8_0.gguf",
        #    "olafgeibig/Nous-Hermes-2-Mistral-7B-DPO-GGUF/nous-hermes-2-mistral-7b-dpo.Q8_0.gguf",
           trust_remote_code=True,  # mandatory for hf models
           max_model_len=20000, gpu_memory_utilization=0.9 
        #    max_new_tokens=128,
        #    top_k=10,
        #    top_p=0.95,
        #    temperature=0.8,
           # tensor_parallel_size=... # for distributed inference
)




tools = [DuckDuckGoSearchRun()]
tool_executor = ToolExecutor(tools=tools)



class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0
    
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reflect and grade the assistant response to the user question below.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_llm_chain = (
    prompt
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
        run_name="Reflection"
    )
    | PydanticToolsParser(tools=[Reflection])
)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    tool_choices = reflection_llm_chain.invoke(inputs)
    print("tool choices", tool_choices)
    reflection = tool_choices[0]
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
    run_name="GenerateInitialCandidate"
)

parser = JsonOutputToolsParser(return_id=True)
# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    res = initial_answer_chain.invoke({"input": state["input"]})
    parsed = parser.invoke(res)
    tool_responses = tool_executor.batch(
        [ToolInvocation(tool=r["type"], tool_input=r["args"]) for r in parsed]
    )
    output_messages = [res] + [
        ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        for resp, tool_call in zip(tool_responses, parsed)
    ]
    reflection = reflection_chain.invoke(
        {"input": state["input"], "candidate": output_messages}
    )
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }

# This generates N candidate values
# for a single input to sample actions from the environment
def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    n = config["configurable"].get("N", 5)
    bound_kwargs = llm.bind_tools(tools=tools).kwargs
    chat_result = llm.generate(
        [messages.to_messages()],
        n=n,
        callbacks=config["callbacks"],
        run_name="GenerateCandidates",
        **bound_kwargs
    )
    return [gen.message for gen in chat_result.generations[0]]


expansion_chain = prompt_template | generate_candidates

# res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
# res 



def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = root.best_child if root.children else root
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    new_candidates = expansion_chain.invoke(
        {"input": state["input"], "messages": messages}, config
    )
    parsed = parser.batch(new_candidates)
    flattened = [
        (i, tool_call)
        for i, tool_calls in enumerate(parsed)
        for tool_call in tool_calls
    ]
    tool_responses = tool_executor.batch(
        [
            ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"])
            for _, tool_call in flattened
        ]
    )
    collected_responses = defaultdict(list)
    for (i, tool_call), resp in zip(flattened, tool_responses):
        collected_responses[i].append(
            ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        )
    output_messages = []
    for i, candidate in enumerate(new_candidates):
        output_messages.append([candidate] + collected_responses[i])

    # Reflect on each candidate
    # For tasks with external validation, you'd add that here.
    reflections = reflection_chain.batch(
        [{"input": state["input"], "candidate": msges} for msges in output_messages],
        config,
    )
    # Grow tree
    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    # We have already extended the tree directly, so we just return the state
    return state

from langgraph.graph import END, StateGraph


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"

def write_file(content:str, output_file_path:str)->None:
    _, file_extension = os.path.splitext(output_file_path)
    file_extension = file_extension.lower()
    try:
        if file_extension in ['.yaml', '.yml']:
            # For YAML, convert the string to a dictionary and save using yaml.dump
            content_dict = {'content': content}
            with open(output_file_path, 'w') as file:
                yaml.dump(content_dict, file, default_flow_style=False)
        elif file_extension in ['.md', '.txt', '.py', '.js', '.ts', '.toml', '.rs']:
            # For Markdown, plain text, Python, JavaScript, TypeScript, TOML, and Rust, save the content directly
            with open(output_file_path, 'w') as file:
                file.write(content)
        elif file_extension == '.json':
            # For JSON, convert the string to a dictionary and save using json.dump
            content_dict = {'content': content}
            with open(output_file_path, 'w') as file:
                json.dump(content_dict, file, indent=4)
        else:
            print(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"An error occurred while saving the file: {str(e)}")

def apply_lats_and_save(prompt:str, filepath:str, modelname:str=""):
    previous_model_name =  os.environ["OPENAI_MODEL_NAME"] 
    if modelname !="":
        os.environ["OPENAI_MODEL_NAME"] = modelname #"gpt-3.5-turbo" #"gpt-4-turbo-preview" #"gpt-4" #"gpt-3.5-turbo"

    builder = StateGraph(TreeState)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.set_entry_point("start")


    builder.add_conditional_edges(
        "start",
        # Either expand/rollout or finish
        should_loop,
    )
    builder.add_conditional_edges(
        "expand",
        # Either continue to rollout or finish
        should_loop,
    )

    graph = builder.compile()

    for step in graph.stream({"input": prompt}):
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")

    solution_node = step["__end__"]["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    print(best_trajectory[-1].content)
    
    if modelname !="":
        os.environ["OPENAI_MODEL_NAME"] = previous_model_name #"gpt-
    write_file(content=best_trajectory[-1].content, output_file_path=filepath )

# question = "Generate a table with the average size and weight, as well as the oldest recorded instance for each of the top 5 most common birds."
question = "Write out magnus carlson series of moves in his game against Alireza Firouzja and propose an alternate strategy"

apply_lats_and_save(question, "/tmp/test.txt")
