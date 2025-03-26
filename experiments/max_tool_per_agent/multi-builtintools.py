from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from termcolor import cprint
import os
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# this script is tested for llama-stack version 0.1.6
"""
How to run:
step 1. ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
step 2. export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
step 3. export LLAMA_STACK_PORT=8321
step 4. llama stack run --image-type conda ~/llama-stack/llama_stack/templates/ollama/run.yaml
    (I'm using conda env, follow this if not using conda, https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html)
step 5. run the example script
"""

# Access the environment variables
inference_model = os.getenv("INFERENCE_MODEL")
tavily_search_api_key = os.getenv("TAVILY_SEARCH_API_KEY")
wolfram_api_key=os.getenv("WOLFRAM_ALPHA_API_KEY")
endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
port = os.getenv("LLAMA_STACK_PORT")
base_url = endpoint if endpoint else f"http://localhost:{port}"
print(f"Model: {inference_model}")

client = LlamaStackClient(
        base_url=base_url,
        provider_data = {"tavily_search_api_key": tavily_search_api_key,
                         "wolfram_alpha_api_key": wolfram_api_key} 
    )  

print(client.toolgroups.list())

# `agent_config` is deprecated. Use inlined parameters instead.
agent = Agent(
    client, 
    model=inference_model,
    instructions="""
    You are a helpful tool calling assistant. 
    There are three builtin tools: websearch, code_interpreter, wolfram_alpha. 
    Select the best tool for each query and execute this query for accurate answer. 
    If there is a error, try another suitable tool.
    """,
    tools=["builtin::websearch",
           "builtin::code_interpreter",
           "builtin::wolfram_alpha"],)
user_prompts = [
    "Hello",
    "Tell me 10 densest elemental metals",
    "solve x^2 + 2x + 1 = 0",
    "Run this code: 2**4 + 8 - 3\nprint(result)",
    "Which teams played in the NBA western conference finals of 2024",
]

session_id = agent.create_session("test-session")
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        session_id=session_id,
    )
    for log in EventLogger().log(response):
        log.print()