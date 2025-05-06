import logging
import re

import torch

from src.nodes.node import LangGraphNode


from langchain_core.prompts import ChatPromptTemplate

from src.agent_states.rewoo_state_schema import AgentState

from langchain.schema.runnable import RunnableLambda




class ReW00Planner(LangGraphNode):

    def __init__(
        self,
        logger: logging,
        local_llm_runnable: RunnableLambda
    ):
        
        super().__init__(logger)

        # config local pipeline for planning
       
        self.regex_patterns = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]" # Regex to match expressions of the form E#... = ...[...]
        
        prompt = self._get_multiple_shot_prompt_planner()
        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])

        self.planner = prompt_template | local_llm_runnable


    def run(self, state: AgentState ) -> AgentState:
        
        task = state["task"]
        result = self.planner.invoke({"task": task})
        # Find all matches in the sample text
        
        matches = re.findall(self.regex_patterns , result.content)

        state['state'] = matches
        state['stplan_stringate'] = result.content

        return state
    

  

    def _get_multiple_shot_prompt_planner(self):
        return """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
                which external tool together with tool input to retrieve evidence. You can store the evidence into a \
                variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

                Tools can be one of the following:
                (1) RETRIEVER[input]: Worker that performs hybrid searches (semantic similarities combined with metadata filtering) 
                over vector databases. Useful when you need to query data about an specific institution. The input should be a search query. The input should be a search query 
                containing the institution name.
                (2) SCRAPPER[input]: Worker that searches results from Google. Useful when you don't find an institution data stored in 
                a vector data base and you need to search it online. This tool must be executed only when no results are retrieved from the vector database.
                The input should be an institution name.
                (3) PERSISTENCE[input]: Worker that persists the data of an institution that has been scrapped from the internet and has not not 
                been persisted in a vector database. This tool must be executed only after an scrapping tool has been executed. 
                The input should be the scrapped text from the internet.
                Useful for making retriever tasks. 
                (4) NER[input]: A pretrained LLM like yourself, specialized in Named Entity Recognition. Useful when you need to retrieve named entities from 
                the data of an institution already stored within the vector database. This tool must be executed only after  


                example 1:
                Task: list the board members of company Nvidia
                Plan: Retrive the data that match the query about Nvidia from the vector database. #E1 = RETRIEVER[Nvidia]
                Plan: List the named entities within the retrieved data from the vector database and consifer this list as the solution #E2 = NER[#E1]

                example 2:
                Task: list the board members of company Bank of America
                Plan: Retrive the data that match the query about Bank of America from the vector database. #E1 = RETRIEVER[Bank of America]
                Plan: List the named entities within the retrieved data from the vector database and consifer this list as the solution #E2 = NER[#E1]
                Plan: If no data is retrieved, collect the Bank of america data from the internet. #E3 = SCRAPPER[Bank of America]
                Plan: Persist the data into the vector database. #E4 = PERSISTENCE[#E3]
                Plan: Retrive the data that match the query about Bank of America from the vector database. #E5 = RETRIEVER[Bank of America]
                Plan: List the named entities within the retrieved data from the vector database and consifer this list as the solution #E6 = NER[#E5]

                Begin! 
                Describe your plans with rich details. Each Plan should be followed by only one #E.

                Task: {task}"""
    

      # in case the plan is bad, consider do this: 1. Try multi-shot prompting. 2. try other model. 3. Fine-tunne this model
    def _get_one_shot_prompt_planner(self):

        return """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
                which external tool together with tool input to retrieve evidence. You can store the evidence into a \
                variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

                Tools can be one of the following:
                (1) RETRIEVER[input]: Worker that performs hybrid searches (semantic similarities combined with metadata filtering) 
                over vector databases. Useful when you need to query data about an specific institution. The input should be a search query. The input should be a search query 
                containing the institution name.
                (2) SCRAPPER[input]: Worker that searches results from Google. Useful when you don't find an institution data stored in 
                a vector data base and you need to search it online. This tool must be executed only when no results are retrieved from the vector database.
                The input should be an institution name.
                (3) PERSISTENCE[input]: Worker that persists the data of an institution that has been scrapped from the internet and has not not 
                been persisted in a vector database. This tool must be executed only after an scrapping tool has been executed. 
                The input should be the scrapped text from the internet.
                Useful for making retriever tasks. 
                (4) NER[input]: A pretrained LLM like yourself, specialized in Named Entity Recognition. Useful when you need to retrieve named entities from 
                the data of an institution already stored within the vector database. This tool must be executed only after  

                example 1:
                Task: list the board members of company Bank of America
                Plan: Retrive the data that match the query about Bank of America from the vector database. #E1 = RETRIEVER[Bank of America]
                Plan: List the named entities within the retrieved data from the vector database and consifer this list as the solution #E2 = NER[#E1]
                Plan: If no data is retrieved, collect the Bank of america data from the internet. #E3 = SCRAPPER[Bank of America]
                Plan: Persist the data into the vector database. #E4 = PERSISTENCE[#E3]
                Plan: Retrive the data that match the query about Bank of America from the vector database. #E5 = RETRIEVER[Bank of America]
                Plan: List the named entities within the retrieved data from the vector database and consifer this list as the solution #E6 = NER[#E5]

                Begin! 
                Describe your plans with rich details. Each Plan should be followed by only one #E.

                Task: {task}"""


    