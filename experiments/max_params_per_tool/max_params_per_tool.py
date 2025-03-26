from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client import LlamaStackClient

from ..tools import ArbitraryClientTool



client = LlamaStackClient(base_url="http://localhost:8321")

for i in range(15,30,5):
    print(i)
    
    arbitrary_client_tool = ArbitraryClientTool(i)
    print(arbitrary_client_tool.get_tool_definition())

    agent_config = AgentConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enable_session_persistence = False,
        instructions = "You are a helpful assistant.",
        toolgroups = [],
        client_tools = [arbitrary_client_tool.get_tool_definition()],
        tool_choice="required",
        tool_prompt_format="json",
        max_infer_iters=4,
        )

    agent = Agent(client=client,
                agent_config=agent_config,
                client_tools=[arbitrary_client_tool]
                )

    session_id = agent.create_session("test")
    response = agent.create_turn(
                messages=[{"role":"user","content":"You must use the arbitrary_client_tool"}],
                session_id= session_id,
                )

    for r in EventLogger().log(response):
        r.print()
    
    print("\n")
