import logging

from src.nodes.node import LangGraphNode

from src.agent_states.logic_state_schema import AgentState

from src.rag_embeddings.vector_database import VectorDB

from src.tools.retriever_tool import RetrieverTool

class Retriever(LangGraphNode):


    def __init__(
        self, 
        logger: logging,
        vector_db: VectorDB 
    ):
        
        super().__init__(logger)
        self.vector_db = vector_db
    

    def run(self, state: AgentState) -> AgentState:

        company_name = state["company_name"]

        self.logger.info(f"Searching for persisted records {company_name}")
        
        tool = RetrieverTool(self.vector_db)
        filtered_docs = tool.invoke_tool(company_name)

        if filtered_docs is not None:
            state["found"] = True
            state["retrieved_text"] = filtered_docs
            self.logger.info(f"Company {company_name} data, already persisted")

        else:
            state["found"] = False

        return state