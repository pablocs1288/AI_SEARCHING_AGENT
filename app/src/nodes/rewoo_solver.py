
import logging

from langchain_core.prompts import ChatPromptTemplate

from src.nodes.node import LangGraphNode
from src.agent_states.rewoo_state_schema import AgentState


from langchain.schema.runnable import RunnableLambda


class ReW00Solver(LangGraphNode):

    def __init__(
        self,
        logger: logging,
        local_llm_runnable: RunnableLambda

    ):
        
        super().__init__(logger)
        
        self.solve_prompt = self._get_solver_prompt()
        
        solver_prompt_template = ChatPromptTemplate.from_messages([("user", self._get_solver_prompt())])
        self.solver = solver_prompt_template | local_llm_runnable
        

    def tool_execution(self, state: AgentState):

        self.logger.info("")
        
        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = (state["results"] or {}) if "results" in state else {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        
        #prompt = self.solve_prompt.format(plan=plan, task=state["task"])
        result = self.solver.invoke({"plan": plan, "task": state["task"]})

        state['result'] = result

        return state
    
    
    def _get_solver_prompt(self):
        return """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
                retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
                contain irrelevant information.

                {plan}

                Now solve the question or task according to provided Evidence above. Respond with the answer
                directly with no extra words.

                Task: {task}
                Response:"""