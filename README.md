# local_llm_langgraph
a repo of examples of using local llm models with langgraph containing two examples. 

* local_llm_langgraph/testing_ollama_llm_w_function_calling.py
    - a mwe of langraph with a single custom tool for evaluating the success of local llms for function calling.  
    - edit the local llm settings for langchain for ollama, vllm, or llama cpp and add the local llm you are testing
    - run `python3 testing_ollama_llm_w_function_calling.py`

* local_llm_langgraph/ollama_llm_lats.py
    - the end goal final example of using a custom reflexion tool and duckduckgo search tool to implment language agent tree search for local llm models
    - to test update the environment variables to for your hosted openai api server/framework and local llm and add a test prompt.
    - run `python3 ollama_llm_lats.py.py`

Note: ollama openai server implmetnation doesn't include function calling / tool use.  Langgraph is missing a bind_tools method for everything other than the default OpenAi `ChatOpenAI`llm method for the lats example
