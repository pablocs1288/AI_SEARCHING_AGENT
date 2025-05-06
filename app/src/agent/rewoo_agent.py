
import os

import logging

from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import Runnable
from langchain.schema.runnable import RunnableLambda

from langchain_core.prompt_values import ChatPromptValue 

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.rag_embeddings.vector_database import VectorDB

from src.agent.base_agent import BaseAgent
from src.agent_states.rewoo_state_schema import AgentState

from src.nodes.rewoo_planner import ReW00Planner
from src.nodes.rewoo_executor import ReW00Executor
from src.nodes.rewoo_solver import ReW00Solver
    


class ReWOOAgent(BaseAgent):

    def __init__(self, logger: logging):
        
        super().__init__(logger)

        # configuring local reasoning models and its execution pipeline (runnable)
        self.local_model, self.local_tokenizer = self._load_model_locally()

        #local_llm_runnable = LocalLLMRunnable(self._local_llm_run)
        local_llm_runnable = RunnableLambda(self._local_llm_run)

        node_planner = ReW00Planner(logger, local_llm_runnable)
        node_executor = ReW00Executor(logger, VectorDB())
        node_solver = ReW00Solver(logger, local_llm_runnable)

        # compiled agent
        self.graph = self._compile_agent(
            planner = node_planner,
            executor =  node_executor,
            solver = node_solver
            )


    def _compile_agent(self, **nodes):

        self.logger.info('Compiling langgraph agent..')
        workflow = StateGraph(state_schema=AgentState)
        workflow.add_node("plan", RunnableLambda(nodes.get('planner').run))
        workflow.add_node("tool", RunnableLambda(nodes.get('executor').run))
        workflow.add_node("solve", RunnableLambda(nodes.get('solver').run))

        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "tool")
        workflow.add_edge("solve", END)
        workflow.add_conditional_edges("tool", RunnableLambda(self._route))

        return workflow.compile()
    
    
    def run_agent(self, company_name: str, verbose: bool= False):
        self.logger.info('Invoking the agent based on ReWOO (reasoning without observation) pattern')
        task = f'list the board members of company {company_name}'

        if verbose: # steam gives detailes of the reasoning, planning an execution
            for s in self.graph.stream({"task": task}):
                self.logger.info(f'[STREAMER] {s}')
            
            result = s['solve']['result'] # the last message, when state already solved the problem
        else:
            result = self.graph.invoke({"task": task}) 
        
        return result


    ##### GRAPH ######
    def _route(self, state):

        if "results" not in state or state["results"] is None:
             _step =  1
        if len(state["results"]) == len(state["steps"]):
             _step =  None
        else:
             _step = len(state["results"]) + 1

        if _step is None:
            return "solve" # We have executed all tasks
        else:
            return "tool" # We are still executing tasks, loop back to the "tool" node
        


    ###### aux methods #####
    def _load_model_locally(self):

        model_id = os.environ['LOCAL_LLM_AGENT']
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        return (AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto"  # Test 'mps' for apple sillicon 
                    ), tokenizer)

    
    def _local_llm_run(self, prompt: ChatPromptValue) -> str:

        pipe = pipeline("text-generation", model=self.local_model, tokenizer=self.local_tokenizer)
        prompt = prompt.to_string()
        result = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.1) # lower temperature more deterministic
        return result[0]['generated_text']
    


    