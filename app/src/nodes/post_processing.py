
import os
import logging

from src.nodes.node import LangGraphNode

from  src.agent_states.logic_state_schema import AgentState

from src.tools.entity_recognition_tool import NERTool


class PostProcessing(LangGraphNode):

    def __init__(
            self, 
            logger: logging 
        ):
        
        super().__init__(logger)
    

    def run(self, state: AgentState) -> AgentState:
        
        company_name = state["company_name"]
        self.logger.info(f"Post-processing board member names of company {company_name}")

        tool = NERTool()
        state["board_members"] = tool.invoke_tool(state["retrieved_text"])

        return state
    