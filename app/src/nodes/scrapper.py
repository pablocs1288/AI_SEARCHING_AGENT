import os
import logging


from src.nodes.node import LangGraphNode

from src.agent_states.logic_state_schema import AgentState

from src.tools.scrapper_tool import ScrapperTool


class Scrapper(LangGraphNode):

    def __init__(
            self, 
            logger: logging 
        ):
        
        super().__init__(logger)
    

    def run(self, state: AgentState) -> AgentState:

        self.logger.info(f"Scrapping web for results (records have not been persisted or there was a problem in the retrieval stage)")
        company_name = state["company_name"]

        tool = ScrapperTool()
        scraped_text = tool.invoke_tool(company_name)

        if not scraped_text:
            state['found'] = True # transition goes to names with retrieve text as the following message where no entities are likely to be recognized
            state["retrieved_text"] = f"No information found online for {company_name}."
        state['scraped_text'] = scraped_text

        return state
