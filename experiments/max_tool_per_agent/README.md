# Evaluating Tool Selection and Scalability in LlamaStack
## Objective
Assess LlamaStack’s ability to handle an increasing number of tools, evaluate tool selection accuracy, and determine whether providing a subset of relevant tools as input improves performance. Also how model series, model size affect performance.
## Key Research Questions
* Scalability: What is the maximum number of tools an agent can handle before performance degrades?
* Tool Selection Accuracy: How well does the agent pick the correct tool for a given query?
* Guided Tool Selection: Does providing a relevant subset of tools in the prompt improve accuracy?
* Model factor: How does model architecture (series, size, temperature type of hyperparameter) affect LlamaStack's tool execution and selection performance?
## Methodology
* Tools:  
  - 3 built-in tools (websearch, wolfram_alpha, code_interpreter).  
  - Dynamically generated N client tools to test scalability (N = 5, 10, 20, 50).
  - also mcp tools (less priority)
* Queries:  
A diverse set of tasks designed to require specific tools.
* **Metrics**:  
  - Scalability Limit (max tool count before performance drops).  
  - Tool Selection Accuracy (% correct, incorrect, or missing tool calls by assert response.steps).  
  - Context size vs tool count. (token size is a strict limitation.)
  - Latency (response time vs. tool count).
  
* Logging:  
Structured logs in CSV format capturing the query, available tools, selected tool, expected tool, execution success, and latency.

Currently, we have multiple built-in tool scripts available. A more systematic evaluation is still a work in progress.

## TODOs
- [ ] Identify and summarize suitable benchmark datasets.
- [ ] Calculate accuracy using appropriate benchmarks.
- [ ] Continue refining metrics and identifying influencing factors.
- [ ] Develop a suitable system prompt wrapper for user queries to ensure the correct tool is executed.


### Test multi-builtin-tool
This is a initial test about having multiple buildin tools configed for one agent. 
The current version works with 0.1.6. If running with previous versions, ensure that `run.yaml` has all three tools configured. 
step 1. `ollama run llama3.2:3b-instruct-fp16 --keepalive 60m`
step 2.
```
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```
step 3: `llama stack run --image-type conda ~/llama-stack/llama_stack/templates/ollama/run.yaml` (I'm using conda env, follow this(https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html) if not using conda)
step 4: run `python multi-builtintools.py`