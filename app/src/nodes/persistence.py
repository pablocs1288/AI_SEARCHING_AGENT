import logging

from src.nodes.node import LangGraphNode

from  src.agent_states.logic_state_schema import AgentState

from src.rag_embeddings.vector_database import VectorDB

from src.tools.persistence_tool import PersistenceTool


class Persistence(LangGraphNode):


    def __init__(
            self,
            logger: logging,  
            vector_db: VectorDB
        ):
        
        super().__init__(logger)
        self.vector_db = vector_db


    def run(self, state: AgentState) -> AgentState:
        
        company_name = state["company_name"]
        self.logger.info(f"Embedding and storing Documents in the vector database of company {company_name}")
        
        text = state["scraped_text"]
        
        tool = PersistenceTool(self.vector_db)
        tool.invoke_tool(text, company_name)
        
        state["retrieved_text"] = text
        return state
