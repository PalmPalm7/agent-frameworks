import os
import uuid
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document

load_dotenv()

# Configuration
LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT", "80")
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

FILE_PATHS = ["documents/cake.rst", 
              "documents/car.rst", 
              "documents/house.rst", 
              "documents/technology.rst"]
VECTOR_DB_PROVIDER = "faiss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 512
TOP_K = 3
MAX_TOKENS_IN_CONTEXT = 4096
MAX_CHUNKS = 2


def create_http_client():
    """Creates an HTTP client for communicating with the Llama Stack server."""
    from llama_stack_client import LlamaStackClient
    endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
    base_url = endpoint if endpoint else f"http://localhost:{LLAMA_STACK_PORT}"
    return LlamaStackClient(
        base_url=base_url
    )


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient
    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


client = create_http_client()  # or create_http_client() depending on the environment

# Load documents for RAG
documents = [
    Document(
        document_id=f"num-{i}",
        content=Path(file_path).read_text(encoding="utf-8"),
        mime_type="text/plain",
        metadata={}
    )
    for i, file_path in enumerate(FILE_PATHS) if Path(file_path).exists()
]

# Print warnings for missing files
missing_files = [file for file in FILE_PATHS if not Path(file).exists()]
if missing_files:
    print(f"Warning: {', '.join(missing_files)} not found. Skipping...")

# Register a vector database
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    provider_id=VECTOR_DB_PROVIDER,
    vector_db_id=vector_db_id,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimension=EMBEDDING_DIMENSION,
)

# Insert documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=CHUNK_SIZE,
)

# Define agent configuration
agent_config = AgentConfig(
    model=INFERENCE_MODEL,
    instructions=(
        "You are a highly reliable, concise, and precise assistant. "
        "When answering user queries, adhere to the following guidelines:\n"
        "1. **Strict Context Usage**: Base your response exclusively on the provided retrieved context.\n"
        "2. **Clarity and Brevity**: Provide clear, focused, and concise responses.\n"
        "3. **Handling Insufficient Information**: If the context lacks sufficient details, state that the necessary information is unavailable.\n"
        "4. **Avoid Inference or Speculation**: Stick to explicit context without speculation.\n"
        "5. **Disambiguation**: If a query is ambiguous, seek clarification before responding.\n"
        "6. **Neutral Tone**: Maintain objectivity in your responses.\n"
        "7. **User-Centric**: Ensure accessibility and clarity for a general audience.\n"
    ),
    enable_session_persistence=False,
    max_infer_iters=2,
    toolgroups=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
                "top_k": TOP_K,
                "query_config": {
                    "max_tokens_in_context": MAX_TOKENS_IN_CONTEXT,
                    "max_chunks": MAX_CHUNKS,
                    "query_generator_config": {
                        "type": "default",
                        "separator": "\n"
                    }
                }
            }
        }
    ],
    tool_choice="auto"
)

rag_agent = Agent(client, agent_config)
session_id = rag_agent.create_session("test-session")

user_prompts = [
    "Give me a recipe to make a cake step by step.",
]

# Run the agent loop
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in EventLogger().log(response):
        log.print()

# Unregister all vector databases
for item in client.vector_dbs.list():
    print(f"Unregistering vector database: {item.identifier}")
    client.vector_dbs.unregister(vector_db_id=item.identifier)
