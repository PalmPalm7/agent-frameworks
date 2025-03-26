import asyncio
import fire
import logging
import os
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@client_tool
def calculator(x: str, y: str, operation: str) -> dict:
    """Simple calculator tool that performs basic math operations.

    :param x: First number to perform operation on
    :param y: Second number to perform operation on
    :param operation: Mathematical operation to perform ('add', 'subtract', 'multiply', 'divide')
    :returns: Dictionary containing success status and result or error message
    """
    logger.debug(f"Calculator called with: x={x}, y={y}, operation={operation}")
    try:
        if operation == "add":
            result = float(x) + float(y)
        elif operation == "subtract":
            result = float(x) - float(y)
        elif operation == "multiply":
            result = float(x) * float(y)
        elif operation == "divide":
            if float(y) == 0:
                return {"success": False, "error": "Cannot divide by zero"}
            result = float(x) / float(y)
        else:
            return {"success": False, "error": "Invalid operation"}

        logger.debug(f"Calculator result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}")
        return {"success": False, "error": str(e)}

async def run_main():
    endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
    port = os.getenv("LLAMA_STACK_PORT")
    base_url = endpoint if endpoint else f"http://localhost:{port}"

    client = LlamaStackClient(base_url=base_url)

    logger.debug("Setting up agent config...")

    agent_config = AgentConfig(
    model=os.getenv("INFERENCE_MODEL"),
    enable_session_persistence = False,
    instructions = """You are a calculator assistant. Use the calculator tool to perform operations.
    When using the calculator tool:
    1. Extract the numbers and operation from the user's request
    2. Use the appropriate operation (add, subtract, multiply, divide)
    3. Present the result clearly
    4. Handle any errors gracefully""",
    toolgroups=[],
    client_tools = [calculator.get_tool_definition()],
    tool_choice="auto",
    # note: the below works for llama-3.1-8B model, but if you plan to use the
    # llama-3.2-3B model, you will need to change it to tool_prompt_format="python_list"
    tool_prompt_format="json",
    )
    logger.debug(f"Agent config: {agent_config}")

    agent = Agent(client=client, 
                agent_config=agent_config,
                client_tools=[calculator],
                )
    session_id = agent.create_session("calc-session")
    logger.debug(f"Created session: {session_id} Agent({agent.agent_id})")

    prompt = "What is 25 plus 15?"
    print(f"\nUser: {prompt}")

    try:
        response_gen = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        event_logger = EventLogger()
        for chunk in event_logger.log(response_gen):
            if chunk is None:
                continue
            try:
                chunk.print(flush=True)
            except Exception as e:
                logger.error(f"Error printing chunk: {e}")
                continue

    except Exception as e:
        logger.error(f"Error during agent turn: {e}")
        raise

def main():
    asyncio.run(run_main())

if __name__ == "__main__":
    fire.Fire(main)
