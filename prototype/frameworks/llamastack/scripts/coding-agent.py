import os
import sys
from termcolor import cprint  # Used for colored terminal output
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient  # Main client for Llama Stack
from llama_stack_client.lib.agents.agent import Agent  # Agent class for AI interactions
from llama_stack_client.lib.agents.event_logger import EventLogger  # Event logging utility
from llama_stack_client.types.agent_create_params import AgentConfig  # Configuration for agent creation
from llama_stack_client.types import Document  # Document type for Llama Stack interactions
import uuid  # For generating unique identifiers

load_dotenv()


def create_http_client():
    """Creates an HTTP client for communicating with the Llama Stack server."""
    from llama_stack_client import LlamaStackClient
    endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
    port = os.getenv("LLAMA_STACK_PORT")
    base_url = endpoint if endpoint else f"http://localhost:{port}"
    return LlamaStackClient(
        base_url=base_url
    )


def create_library_client(template="ollama"):
    """Creates a library-based client for Llama Stack."""
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    if not client.initialize():
        print("Llama Stack not built properly")
        sys.exit(1)
    return client


# Initialize the Llama Stack client, choosing between library or HTTP client
client = create_http_client()  # Switch to create_library_client() if needed

# Define agent configuration
agent_config = AgentConfig(
    model=os.getenv("INFERENCE_MODEL"),  # Fetch model from environment variable
    instructions=(
        "You are a highly reliable, concise, and precise assistant always show the generated code presented to you never generate your own code and never anticipate the result. "
        "When answering user queries, adhere to the following guidelines:\n"
        "1. **Strict Context Usage**: Base your response exclusively on the provided retrieved context.\n"
        "2. **Clarity and Brevity**: Provide clear, focused, and concise responses.\n"
        "3. **Handling Insufficient Information**: If the context lacks sufficient details, state that the necessary information is unavailable.\n"
        "4. **Avoid Inference or Speculation**: Stick to explicit context without speculation.\n"
        "5. **Disambiguation**: If a query is ambiguous, seek clarification before responding.\n"
        "6. **Neutral Tone**: Maintain objectivity in your responses.\n"
        "7. **User-Centric**: Ensure accessibility and clarity for a general audience.\n"
    ),  # General AI instructions
    
    # Enable tool usage for additional functionalities (e.g., RAG, code execution)
    toolgroups=[
        "builtin::code_interpreter",  # Enables execution of code within the AI
    ],
    
    max_infer_iters=5,  # Maximum inference iterations per response
    
    sampling_params={  # Defines sampling strategy for AI response generation
        "strategy": {"type": "greedy", "temperature": 0.7, "top_p": 0.95},
        "max_tokens": 2048,  # Token limit per response
    },
    enable_session_persistence=True  # Ensures session persistence across interactions
)

# Create an agent instance with the configured settings
agent = Agent(client, agent_config)

# Start a monitored session
session_id = agent.create_session("monitored_session")

# Send a user query to the AI agent
response = agent.create_turn(
    messages=[{"role": "user", 
               "content": """Run this code: 2**4 + 8 - 3\nprint(result)")"""
               }],
    session_id=session_id,  # Ensure session ID is correctly passed
)

# Monitor and log each execution step
for log in EventLogger().log(response):
    log.print()
