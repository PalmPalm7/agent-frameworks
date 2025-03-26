import os
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client import LlamaStackClient
from termcolor import cprint

## Start the llama stack server with vLLM backend

# export LLAMA_STACK_PORT=8321
# podman run \
#   -it \
#   -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
#   -v ./run.yaml:/root/my-run.yaml \
#   llamastack/distribution-remote-vllm \
#   --yaml-config /root/my-run.yaml \
#   --port $LLAMA_STACK_PORT \
#   --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
#   --env VLLM_URL=xxx

## Start the local MCP server
# git clone https://github.com/modelcontextprotocol/python-sdk
# Follow instructions to get the env ready
# cd examples/servers/simple-tool
# export MCP_PORT=8000
# uv run mcp-simple-tool --transport sse --port $MCP_PORT

# Connect to the llama stack server
llama_stack_port = os.environ.get("LLAMA_STACK_PORT", "8321")
base_url=f"http://localhost:{llama_stack_port}"
model_id="meta-llama/Llama-3.2-3B-Instruct"
client = LlamaStackClient(base_url=base_url)


# Register MCP tools
# If you're running llamastack through conda 
# then http://localhost:8000/sse will work for mcp_endpoint
mcp_port = os.environ.get("MCP_PORT", "8000")
client.toolgroups.register(
    toolgroup_id="mcp::filesystem",
    provider_id="model-context-protocol",
    mcp_endpoint={"uri":f"http://host.containers.internal:{mcp_port}/sse"})

# Define an agent with MCP toolgroup 
agent_config = AgentConfig(
    model=model_id,
    instructions="You are a helpful assistant",
    toolgroups=["mcp::filesystem"],
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
)
agent = Agent(client, agent_config)
user_prompts = [
    "Fetch content from https://www.google.com and summarize the response"
]

# Run a session with the agent
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
