
import logging

from src.rag_embeddings.vector_database import VectorDB

from src.nodes.node import LangGraphNode
from src.agent_states.rewoo_state_schema import AgentState

from src.tools.entity_recognition_tool import NERTool
from src.tools.persistence_tool import PersistenceTool
from src.tools.retriever_tool import RetrieverTool
from src.tools.scrapper_tool import ScrapperTool

class ReW00Executor(LangGraphNode):

    def __init__(
        self,
        logger: logging,
        vector_db: VectorDB
    ):
        
        super().__init__(logger)

        self.tool_ner = NERTool()
        self.tool_persistence = PersistenceTool(vector_db)
        self.tool_retriver = RetrieverTool(vector_db)
        self.tool_scrapper = ScrapperTool()


    def tool_execution(self, state: AgentState):
        """Worker node that executes the tools of a given plan."""

        _step = self._get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        
        _results = (state["results"] or {}) if "results" in state else {}
        
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        
        self.logger.info(f"[TOOL NODE] Executing {tool} tool")

        if tool == "PERSISTENCE":
            result = self.tool_persistence.invoke_tool(tool_input)
        elif tool == "NER":
            result = self.tool_ner.invoke_tool(tool_input)
        elif tool == "SCRAPPER":
            result = self.tool_scrapper.invoke_tool(tool_input)
        elif tool == "RETRIEVER":
            result = self.tool_retriver.invoke_tool(tool_input)
        else:
            raise ValueError
        
        _results[step_name] = str(result)
        return {"results": _results}
    
    def _get_current_task(self, state: AgentState):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1
        
   