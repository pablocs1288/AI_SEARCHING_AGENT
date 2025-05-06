import logging


from src.agent_states.logic_state_schema import AgentState


class BaseAgent:

    def __init__(self, logger: logging):
        self.logger = logger


    def _compile_agent(self, **nodes):
        pass

    def run_agent(self, initial_state: dict):
        final_state = initial_state
        return final_state