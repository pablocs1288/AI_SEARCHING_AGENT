from typing import TypedDict, List

class AgentState(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str