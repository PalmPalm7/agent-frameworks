from llama_stack_client.lib.agents.client_tool import ClientTool
from llama_stack_client.types.tool_def_param import Parameter


class ArbitraryClientTool(ClientTool):
    def __init__(self, n):
        self.n = n
    
    def _arbitrary_tool(self, *kwargs):
        return kwargs

    def _generate_kwargs(self, num:int) -> dict:
       kwargs = {f"query_{n}": Parameter(
                name=f"query_{n}",
                parameter_type="str",
                description=f"query_{n}",
                required=True,
            ) for n in range(num)}
       
       return kwargs

    def get_name(self):
        return "arbitrary_tool"
    
    def get_description(self):
        return "This tool is used to evaluate the number of parameters an LLM can manage during tool calling."
    
    def get_params_definition(self):
        return self._generate_kwargs(self.n)
    
    def run_impl(self, **kwargs):
        return self._arbitrary_tool(kwargs)
