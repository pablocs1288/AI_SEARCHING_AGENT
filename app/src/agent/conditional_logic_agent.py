
import logging

from langgraph.graph import END, StateGraph
from langchain.schema.runnable import RunnableLambda

from src.agent.base_agent import BaseAgent
from src.agent_states.logic_state_schema import AgentState

from src.rag_embeddings.vector_database import VectorDB

from src.nodes.persistence import Persistence
from src.nodes.retriever import Retriever
from src.nodes.post_processing import PostProcessing
from src.nodes.scrapper import Scrapper

class LogicAgent(BaseAgent):

    def __init__(self, logger: logging):

        super().__init__(logger)
        vectordb = VectorDB()
        
        node_persistence = Persistence(logger, vectordb)
        node_retriever = Retriever(logger, vectordb)
        node_post_processing = PostProcessing(logger)
        node_scrapper = Scrapper(logger)

       
        # compiled agent
        self.graph = self._compile_agent(
            persistence = node_persistence,
            retriever =  node_retriever,
            post_processing = node_post_processing,
            scrapper =  node_scrapper,
            )

    def _compile_agent(self, **nodes):

        self.logger.info('Compiling langgraph agent..')
        workflow = StateGraph(state_schema=AgentState)
        workflow.add_node("CheckDB", RunnableLambda(nodes.get('retriever').run))
        workflow.add_node("SearchWeb", RunnableLambda(nodes.get('scrapper').run))
        workflow.add_node("EmbedAndStore", RunnableLambda(nodes.get('persistence').run))
        workflow.add_node("ReturnNames", RunnableLambda(nodes.get('post_processing').run))

        workflow.set_entry_point("CheckDB")

        workflow.add_conditional_edges("CheckDB", lambda state: "ReturnNames" if state["found"] else "SearchWeb")
        workflow.add_edge("SearchWeb", "EmbedAndStore")
        workflow.add_edge("EmbedAndStore", "ReturnNames")
        workflow.add_edge("ReturnNames", END)

        return workflow.compile()


    def run_agent(self, company_name: str):
        initial_state = {"company_name": company_name}
        self.logger.info('Invoking the agent based on conditional logic reasoning pattern')
        final_state = self.graph.invoke(initial_state)
        return final_state["board_members"]


