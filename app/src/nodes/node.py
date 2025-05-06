import logging


from  src.agent_states.logic_state_schema import AgentState


class LangGraphNode:

    def __init__(self, logger: logging):
        self.logger = logger


    def run(self, state: AgentState) -> AgentState:
        return state